# -*- coding: utf-8 -*-
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
# 第0部分: 环境设置
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set to {seed}")


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sns.set_theme(style="whitegrid")
output_dir = "output_optimized_cosine_noisy_oracle"  # 修改输出目录名以区分
os.makedirs(output_dir, exist_ok=True)


# 绘图辅助函数 (保持不变)
def plot_logo(one_hot_seqs, title=''):
    if isinstance(one_hot_seqs, torch.Tensor):
        one_hot_seqs = one_hot_seqs.detach().cpu().numpy()
    if one_hot_seqs.shape[-1] != 4:
        one_hot_seqs = np.transpose(one_hot_seqs, (0, 2, 1))
    pwm = np.mean(one_hot_seqs, axis=0)
    pwm_df = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])
    fig, ax = plt.subplots(1, 1, figsize=(10, 2.5))
    logo = logomaker.Logo(pwm_df, ax=ax, shade_below=0.5, fade_below=0.5, font_name='sans-serif')
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    if title: ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"logo_{title.replace(' ', '_')}.png"), dpi=300)
    plt.show()


# ==============================================================================
# 第1部分: PyTorch 模型定义
# ==============================================================================
SEQUENCE_LENGTH = 17
NUCLEOTIDES = 4
CSV_FILE = "rbs_data_f.csv"  # 请确保文件路径正确


# --- UNet 组件 (保持不变) ---
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embed = np.log(10000) / (half_dim - 1)
        embed = torch.exp(torch.arange(half_dim, device=device) * -embed)
        embed = t[:, None] * embed[None, :]
        embed = torch.cat((embed.sin(), embed.cos()), dim=1)
        return embed


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.mlp_time = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = F.silu(self.norm1(self.conv1(x)))
        h += self.mlp_time(t)[:, :, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):  # 简化版 Self-Attention
    def __init__(self, n_channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, n_channels)
        self.qkv = nn.Conv1d(n_channels, n_channels * 3, 1)
        self.out = nn.Conv1d(n_channels, n_channels, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.qkv(self.norm(x)).view(b, 3, c, l)
        q, k, v = qkv.unbind(1)
        attn = torch.einsum('bcl,bck->blk', q, k) * (c ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('blk,bck->bcl', attn, v)
        return x + self.out(out)


class UpDownBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim, down=True):
        super().__init__()
        self.res = ResidualBlock(in_c, out_c, time_dim)
        self.attn = AttentionBlock(out_c)
        self.sample = nn.Conv1d(out_c, out_c, 3, stride=2, padding=1) if down else nn.ConvTranspose1d(out_c, out_c, 4,
                                                                                                      stride=2,
                                                                                                      padding=1)

    def forward(self, x, t):
        x = self.attn(self.res(x, t))
        return self.sample(x)


class UNet(nn.Module):
    def __init__(self, n_channels=128):  # 保持优化后的 128
        super().__init__()
        time_dim = n_channels * 4
        self.time_embedding = nn.Sequential(TimeEmbedding(n_channels), nn.Linear(n_channels, time_dim), nn.SiLU(),
                                            nn.Linear(time_dim, time_dim))
        self.in_conv = nn.Conv1d(NUCLEOTIDES, n_channels, 3, padding=1)
        self.down1 = UpDownBlock(n_channels, n_channels * 2, time_dim, True)
        self.mid_res1 = ResidualBlock(n_channels * 2, n_channels * 2, time_dim)
        self.mid_attn = AttentionBlock(n_channels * 2)
        self.mid_res2 = ResidualBlock(n_channels * 2, n_channels * 2, time_dim)
        self.up1 = UpDownBlock(n_channels * 4, n_channels, time_dim, False)  # Skip connection concat
        self.out_res = ResidualBlock(n_channels * 2, n_channels, time_dim)
        self.out_conv = nn.Conv1d(n_channels, NUCLEOTIDES, 1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        x_in = self.in_conv(x)
        x_d1 = self.down1(x_in, t_emb)
        x_mid = self.mid_res2(self.mid_attn(self.mid_res1(x_d1, t_emb)), t_emb)
        # Up-sample
        x_u1 = self.up1(torch.cat([x_mid, x_d1], dim=1), t_emb)
        # Handle potential shape mismatch due to padding/stride
        if x_u1.shape[2] != x_in.shape[2]:
            x_u1 = F.interpolate(x_u1, size=x_in.shape[2], mode='linear')
        x_out = self.out_res(torch.cat([x_u1, x_in], dim=1), t_emb)
        return self.out_conv(x_out)


# --- Oracle 模型 (保持不变) ---
class OracleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(NUCLEOTIDES, 128, 7, padding='same'), nn.ReLU(),
            nn.Dropout(0.3),  # 保持优化后的 0.3
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, padding='same'), nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        self.fc_net = nn.Sequential(nn.Linear(256 * (SEQUENCE_LENGTH // 4), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, seq): return self.fc_net(self.conv_net(seq))

    def get_embedding(self, seq): return self.fc_net[0](self.conv_net(seq))


# --- Diffusion 类 (【重要修改】：方案一 Cosine Schedule) ---
class Diffusion:
    def __init__(self, noise_predictor_model, T=1000):
        self.model = noise_predictor_model
        self.T = T

        # [修改点 1] 使用 Cosine Noise Schedule
        # 相比 Linear，Cosine 在 t 较小时保留更多信号，更适合离散数据
        s = 0.008
        steps = T + 1
        x = torch.linspace(0, T, steps, device=device)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(betas, 0.0001, 0.9999)  # 防止数值不稳定

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def q_sample(self, x_0, t, noise=None):
        if noise is None: noise = torch.randn_like(x_0)
        mean = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1) * x_0
        std = torch.sqrt(1. - self.alphas_cumprod[t]).view(-1, 1, 1)
        return mean + std * noise

    def train_step(self, x_0):
        t = torch.randint(0, self.T, (x_0.shape[0],), device=device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t.float())
        return F.mse_loss(predicted_noise, noise)


# ==============================================================================
# 第2, 3, 4部分: 数据与训练
# ==============================================================================
print("--- Part 2: Loading Data ---")
# 模拟数据加载 (请确保本地有 csv 文件)
if not os.path.exists(CSV_FILE):
    print("Warning: CSV not found, generating dummy data for demo...")
    sequences = ["".join(np.random.choice(['A', 'C', 'G', 'T'], 17)) for _ in range(1000)]
    rl_scores = np.random.rand(1000).astype(np.float32)
else:
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


    def one_hot_encode(seq):
        if len(seq) != SEQUENCE_LENGTH: return None
        encoded = np.zeros((NUCLEOTIDES, SEQUENCE_LENGTH), dtype=np.float32)
        for i, char in enumerate(seq.upper()):
            if char in char_map:
                encoded[char_map[char], i] = 1.0
            else:
                return None
        return encoded


    df = pd.read_csv(CSV_FILE)
    df_subset = df.head(310000)
    processed_data = [(one_hot_encode(s), r) for s, r in zip(df_subset['序列'], df_subset['rl']) if
                      one_hot_encode(s) is not None]
    sequences = [item[0] for item in processed_data]
    rl_scores = np.array([item[1] for item in processed_data])

one_hot_sequences = np.array(sequences).astype(np.float32)
scores = rl_scores.astype(np.float32)
print(f"Data loaded. Shape: {one_hot_sequences.shape}")

# --- [修改点 2] Oracle 训练 (方案二：加噪训练) ---
print("\n--- Part 3: Training Robust Oracle (with Noise) ---")
X_train, X_val, y_train, y_val = train_test_split(one_hot_sequences, scores, test_size=0.2, random_state=42)
oracle_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32,
                                 shuffle=True)

real_oracle = OracleCNN().to(device)
oracle_optimizer = optim.Adam(real_oracle.parameters(), lr=0.0005)  # 保持优化后的 LR
criterion = nn.MSELoss()

ORACLE_EPOCHS = 50
for epoch in range(ORACLE_EPOCHS):
    real_oracle.train()
    total_loss = 0
    for seqs, labels in oracle_train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        oracle_optimizer.zero_grad()

        # [关键修改]: 在输入 Oracle 前加入随机高斯噪声
        # 模拟扩散过程中的中间状态，提高 Oracle 对模糊输入的鲁棒性
        noise_level = torch.rand(seqs.shape[0], 1, 1).to(device) * 0.2  # 0到0.2强度的噪声
        noise = torch.randn_like(seqs) * noise_level
        noisy_input = seqs + noise

        preds = real_oracle(noisy_input).squeeze()
        loss = criterion(preds, labels)

        loss.backward()
        oracle_optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Oracle Epoch {epoch + 1}/{ORACLE_EPOCHS}, Loss: {total_loss / len(oracle_train_loader):.4f}")

real_oracle.eval()

# --- Diffusion 训练 (保持 T=1000, 128 Channels) ---
print("\n--- Part 4: Training Diffusion ---")
diffusion_train_loader = DataLoader(TensorDataset(torch.from_numpy(one_hot_sequences)), batch_size=64, shuffle=True)
unet = UNet(n_channels=128).to(device)
diffusion = Diffusion(unet, T=1000)  # 使用更新后的 Cosine Schedule 类
diffusion_optimizer = optim.Adam(unet.parameters(), lr=1e-4)

DIFFUSION_EPOCHS = 100
for epoch in range(DIFFUSION_EPOCHS):
    unet.train()
    epoch_loss = 0
    pbar = tqdm(diffusion_train_loader, desc=f"Diff Epoch {epoch + 1}", leave=False)
    for (real_seqs,) in pbar:
        diffusion_optimizer.zero_grad()
        loss = diffusion.train_step(real_seqs.to(device))
        loss.backward()
        diffusion_optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

print("Diffusion training finished.")


# ==============================================================================
# 第5部分: 引导采样函数 (针对 Cosine Schedule 微调参数)
# ==============================================================================
def guided_sampling(diffusion_model, oracle_model, batch_size, guidance_scale=1.0):
    x_t = torch.randn((batch_size, NUCLEOTIDES, SEQUENCE_LENGTH), device=device)

    for t in tqdm(reversed(range(diffusion_model.T)), desc="Sampling", total=diffusion_model.T, leave=False):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # 计算引导梯度
        grad = torch.zeros_like(x_t)
        if guidance_scale > 0:
            with torch.enable_grad():
                x_in = x_t.detach().requires_grad_()
                # [小技巧] 缩放输入，使其在数值上更接近 Oracle 训练时的分布（虽然 Oracle 已经加噪训练过）
                scores = oracle_model(x_in).sum()
                grad = torch.autograd.grad(scores, x_in)[0]
                # 梯度截断，防止单步更新过大
                grad = torch.clamp(grad, -0.1, 0.1)

        with torch.no_grad():
            predicted_noise = diffusion_model.model(x_t, t_tensor.float())

            # 获取当前步的 alpha, beta
            alpha_t = diffusion_model.alphas[t]
            alpha_t_cumprod = diffusion_model.alphas_cumprod[t]
            beta_t = diffusion_model.betas[t]

            # 加入引导项：修改预测的噪声
            # guided_noise = pred_noise - sqrt(1 - alpha_bar) * gradient * scale
            guided_noise = predicted_noise - torch.sqrt(1. - alpha_t_cumprod) * grad * guidance_scale

            # DDPM 采样公式
            # mean = (1 / sqrt(alpha)) * (x_t - (beta / sqrt(1 - alpha_bar)) * noise)
            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = beta_t / torch.sqrt(1 - alpha_t_cumprod)
            mean = coeff1 * (x_t - coeff2 * guided_noise)

            if t > 0:
                # 这里的 variance 取 beta_t 即可 (DDPM 论文标准)
                noise = torch.randn_like(x_t)
                sigma = torch.sqrt(beta_t)
                x_t = mean + sigma * noise
            else:
                x_t = mean

    # 最终输出 One-hot
    return F.one_hot(torch.argmax(x_t, dim=1), num_classes=NUCLEOTIDES).float().permute(0, 2, 1)


# ==============================================================================
# 第6部分: 生成评估
# ==============================================================================
print("\n--- Part 6: Evaluation ---")
# 生成 200 条 RL 引导的序列
gen_seqs_onehot = guided_sampling(diffusion, real_oracle, batch_size=32, guidance_scale=2.0)  # 尝试提高 guidance 到 2.0
gen_seqs_onehot_tensor = gen_seqs_onehot  # Keep tensor for oracle

# 转换回字符串
idx_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


def to_str(oh):
    indices = torch.argmax(oh, dim=1).cpu().numpy()
    return ["".join([idx_to_char[i] for i in s]) for s in indices]


gen_strs = to_str(gen_seqs_onehot)

# 评估分数
with torch.no_grad():
    gen_scores = real_oracle(gen_seqs_onehot).cpu().numpy().flatten()

print(f"Generated {len(gen_strs)} sequences.")
print(f"Average Predicted Score: {np.mean(gen_scores):.4f}")
print(f"Top 5 Scores: {sorted(gen_scores, reverse=True)[:5]}")

# 绘制 Top 序列的 Logo
top_indices = np.argsort(gen_scores)[-20:]  # Top 20
plot_logo(gen_seqs_onehot[top_indices], title="Top 20 Generated Sequences (Cosine + NoisyOracle)")

print("Script finished.")