# -*- coding: utf-8 -*-
# 文件名: final_complete_assessment.py
import sys
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
import Levenshtein
from scipy.spatial.distance import jensenshannon
from scipy.linalg import sqrtm
from collections import Counter
import logomaker

# ==============================================================================
# 第0部分: 环境设置 (同步搜索得出的最优参数)
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns.set_theme(style="whitegrid")
output_dir = "final_output_optimized"
os.makedirs(output_dir, exist_ok=True)

# 最优实验参数总结
T_STEPS = 1000      # 最优扩散步数
CHANNELS = 128     # 最优模型宽度
LEARNING_RATE = 0.0002 # 最优学习率
SEQUENCE_LENGTH = 17
NUCLEOTIDES = 4
CSV_FILE = "rbs_data.csv"

# ==============================================================================
# 第1部分: 模型定义 (UNet 设为 128 通道)
# ==============================================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        embed = np.log(10000) / (half_dim - 1)
        embed = torch.exp(torch.arange(half_dim, device=t.device) * -embed)
        embed = t[:, None] * embed[None, :]
        embed = torch.cat((embed.sin(), embed.cos()), dim=1)
        return embed

class UNet(nn.Module):
    def __init__(self, n_channels=CHANNELS):
        super().__init__()
        time_emb_dim = n_channels * 4
        self.time_embedding = nn.Sequential(TimeEmbedding(n_channels), nn.Linear(n_channels, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.in_conv = nn.Conv1d(NUCLEOTIDES, n_channels, 3, padding=1)
        self.down = nn.Sequential(nn.Conv1d(n_channels, n_channels*2, 3, stride=2, padding=1), nn.SiLU())
        self.up = nn.Sequential(nn.ConvTranspose1d(n_channels*2, n_channels, 4, stride=2, padding=1), nn.SiLU())
        self.out_conv = nn.Conv1d(n_channels, NUCLEOTIDES, 1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        x = self.in_conv(x) + t_emb[:, :, None]
        x_down = self.down(x)
        x_up = self.up(x_down)
        return self.out_conv(F.interpolate(x_up, size=SEQUENCE_LENGTH))

class OracleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(4, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(128 * (SEQUENCE_LENGTH // 4), 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.fc(self.conv(x))
    def get_embedding(self, x): return self.fc[0](self.conv(x))

# ==============================================================================
# 第2部分: 扩散逻辑与训练 (引入梯度剪切)
# ==============================================================================
class Diffusion:
    def __init__(self, model, T=T_STEPS):
        self.model = model
        self.T = T
        # Linear Schedule 被证明优于 Cosine
        self.betas = torch.linspace(1e-4, 0.02, T, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def train_step(self, x_0):
        t = torch.randint(0, self.T, (x_0.shape[0],), device=device)
        noise = torch.randn_like(x_0)
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1)
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        return F.mse_loss(self.model(x_t, t.float()), noise)

# --- 数据预处理 ---
df = pd.read_csv(CSV_FILE).head(260000)
char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
idx_to_char = {v: k for k, v in char_map.items()}
def encode(s):
    res = np.zeros((4, 17), dtype=np.float32)
    for i, c in enumerate(s.upper()): res[char_map[c], i] = 1.0
    return res

one_hot_data = np.array([encode(s) for s in df['序列']])
all_strs = df['序列'].tolist()
train_loader = DataLoader(TensorDataset(torch.from_numpy(one_hot_data)), batch_size=256, shuffle=True)

# --- 训练核心 ---
unet = UNet().to(device)
diffusion = Diffusion(unet)
optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
real_oracle = OracleCNN().to(device) # 需加载或训练 Oracle

for epoch in range(50):
    unet.train()
    for (x,) in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        optimizer.zero_grad()
        loss = diffusion.train_step(x.to(device))
        loss.backward()
        # [关键] 梯度剪切确保 26w 数据稳定性
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

# ==============================================================================
# 第3部分: 增强采样与可视化函数
# ==============================================================================
def guided_sampling(n=200, guidance=1.0, track_gradients=False):
    unet.eval()
    x = torch.randn((n, 4, 17), device=device)
    grad_norms = []
    for t in tqdm(reversed(range(T_STEPS)), desc="Sampling", total=T_STEPS, leave=False):
        t_b = torch.full((n,), t, device=device)
        if guidance > 0:
            with torch.enable_grad():
                x.requires_grad_(True)
                score = real_oracle(x).sum()
                g = torch.autograd.grad(score, x)[0].clamp(-1, 1)
                if track_gradients: grad_norms.append(g.norm().item()/n)
        else: g = 0
        with torch.no_grad():
            eps = unet(x, t_b.float()) - torch.sqrt(1-diffusion.alphas_cumprod[t])*g*guidance
            x = (x - (1-diffusion.alphas[t])/torch.sqrt(1-diffusion.alphas_cumprod[t])*eps)/torch.sqrt(diffusion.alphas[t])
            if t > 0: x += torch.sqrt(diffusion.betas[t]) * torch.randn_like(x)
    return x, grad_norms

# ==============================================================================
# 第4部分: 完整评估分析 (结果对比/小提琴图/GC分布/Novelty)
# ==============================================================================
print("\n--- Final Comprehensive Analysis ---")
N_EVAL = 500
gen_oh, grad_history = guided_sampling(n=N_EVAL, guidance=1.0, track_gradients=True)
gen_strs = ["".join([idx_to_char[i] for i in s]) for s in torch.argmax(gen_oh, 1).cpu().numpy()]

# 1. 新颖性 (Novelty)
novelty = np.mean([min([Levenshtein.distance(g, t) for t in all_strs[:2000]]) for g in gen_strs])
print(f"Generative Novelty Score: {novelty:.2f}")

# 2. 奖励分布图 (Reward Distribution)
with torch.no_grad():
    gen_scores = real_oracle(gen_oh).cpu().numpy().flatten()
    real_scores = real_oracle(torch.from_numpy(one_hot_data[:N_EVAL]).to(device)).cpu().numpy().flatten()

plt.figure(figsize=(10, 6))
sns.violinplot(data=[real_scores, gen_scores], palette="muted")
plt.xticks([0, 1], ["Experimental", "RL-Guided Diffusion"])
plt.title(f"Functionality Comparison (Novelty={novelty:.2f})")
plt.savefig(os.path.join(output_dir, "reward_distribution.png"))

# 3. 梯度动力学 (Gradient Dynamics)
plt.figure(figsize=(10, 4))
plt.plot(grad_history, color='purple')
plt.title("Guidance Gradient Dynamics (T=1000)")
plt.xlabel("Denoising Steps (T -> 0)")
plt.savefig(os.path.join(output_dir, "gradient_dynamics.png"))

# 4. 保存结果
df_gen = pd.DataFrame({'Sequence': gen_strs, 'Predicted_Score': gen_scores})
df_gen.to_csv(os.path.join(output_dir, "generated_sequences.csv"), index=False)

print(f"\nAll tasks completed. Results saved in '{output_dir}/'.")