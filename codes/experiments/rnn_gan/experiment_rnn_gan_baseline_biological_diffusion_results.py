# -*- coding: utf-8 -*-
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
from sklearn.manifold import TSNE
import logomaker
from tqdm import tqdm


# ==============================================================================
# 第0部分: 基础配置
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "biological_diffusion_final_results"
os.makedirs(output_dir, exist_ok=True)

SEQUENCE_LENGTH = 17
NUCLEOTIDES = 4
CSV_FILE = "rbs_data.csv"
char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
idx_to_char = {v: k for k, v in char_map.items()}


# ==============================================================================
# 第1部分: 数据加载与预处理 (补全代码)
# ==============================================================================
def one_hot_encode(seq):
    if len(seq) != SEQUENCE_LENGTH: return None
    encoded = np.zeros((4, SEQUENCE_LENGTH), dtype=np.float32)
    for i, char in enumerate(seq.upper()):
        if char in char_map:
            encoded[char_map[char], i] = 1.0
        else:
            return None
    return encoded


def one_hot_to_strings(oh_tensor):
    indices = torch.argmax(oh_tensor, dim=1).cpu().numpy()
    return ["".join([idx_to_char[i] for i in seq]) for seq in indices]


print("--- Step 1: Loading Data ---")
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} sequences from {CSV_FILE}")
else:
    print("CSV not found, generating dummy data...")
    df = pd.DataFrame({
        '序列': ["".join(random.choices("ACGT", k=17)) for _ in range(2000)],
        'rl': np.random.rand(2000)
    })

# 转换数据
processed_data = []
for s, r in zip(df['序列'].values, df['rl'].values):
    oh = one_hot_encode(str(s))
    if oh is not None:
        processed_data.append((oh, float(r)))

all_oh = np.array([x[0] for x in processed_data])
all_scores = np.array([x[1] for x in processed_data])
print(f"Final dataset size: {len(all_oh)}")


# ==============================================================================
# 第2部分: 模型架构
# ==============================================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=1)


import math


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, t_dim):
        super().__init__()
        self.mlp_t = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_c))
        self.conv1 = nn.Conv1d(in_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t):
        h = self.act(self.norm1(self.conv1(x)))
        h += self.mlp_t(t)[:, :, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, n_channels=64):
        super().__init__()
        t_dim = n_channels * 4
        self.time_emb = nn.Sequential(TimeEmbedding(n_channels), nn.Linear(n_channels, t_dim), nn.SiLU(),
                                      nn.Linear(t_dim, t_dim))
        self.in_conv = nn.Conv1d(4, n_channels, 3, padding=1)
        self.d1 = ResidualBlock(n_channels, n_channels * 2, t_dim)
        self.d2 = ResidualBlock(n_channels * 2, n_channels * 4, t_dim)
        self.u1 = ResidualBlock(n_channels * 4, n_channels * 2, t_dim)
        self.u2 = ResidualBlock(n_channels * 4, n_channels, t_dim)
        self.out = nn.Conv1d(n_channels, 4, 1)

    def forward(self, x, t):
        t_enc = self.time_emb(t)
        x0 = self.in_conv(x)
        x1 = self.d1(x0, t_enc)
        x2 = self.d2(x1, t_enc)
        # 简单Up层
        up1 = self.u1(x2, t_enc)
        up2 = self.u2(torch.cat([F.interpolate(up1, x1.shape[2]), x1], dim=1), t_enc)
        return self.out(up2)


class OracleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(4, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2))
        self.fc = nn.Sequential(nn.Linear(64 * 8, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x): return self.fc(self.conv(x).flatten(1))

    def get_embedding(self, x): return self.fc[0](self.conv(x).flatten(1))


# ==============================================================================
# 第3部分: 训练逻辑 (补全代码)
# ==============================================================================
print("--- Step 2: Training Oracle (Predictor) ---")
real_oracle = OracleCNN().to(device)
opt_o = optim.Adam(real_oracle.parameters(), lr=1e-3)
for epoch in range(10):
    real_oracle.train()
    total_loss = 0
    for i in range(0, len(all_oh), 32):
        b_x = torch.from_numpy(all_oh[i:i + 32]).to(device)
        b_y = torch.from_numpy(all_scores[i:i + 32]).to(device)
        opt_o.zero_grad()
        loss = F.mse_loss(real_oracle(b_x).squeeze(), b_y)
        loss.backward()
        opt_o.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0: print(f"Oracle Epoch {epoch + 1} Loss: {total_loss / len(all_oh):.6f}")

print("--- Step 3: Training Diffusion Generator ---")
unet = UNet().to(device)
betas = torch.linspace(1e-4, 0.02, 500, device=device)
alphas_cp = torch.cumprod(1. - betas, dim=0)
opt_d = optim.Adam(unet.parameters(), lr=1e-4)

for epoch in tqdm(range(20), desc="Diffusion Training"):
    for i in range(0, len(all_oh), 64):
        x_0 = torch.from_numpy(all_oh[i:i + 64]).to(device)
        t = torch.randint(0, 500, (x_0.shape[0],), device=device)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alphas_cp[t]).view(-1, 1, 1) * x_0 + torch.sqrt(1. - alphas_cp[t]).view(-1, 1, 1) * noise
        opt_d.zero_grad()
        loss = F.mse_loss(unet(x_t, t.float()), noise)
        loss.backward()
        opt_d.step()


# ==============================================================================
# 第4部分: 采样与结果生成
# ==============================================================================
def sample(unet, oracle, scale=1.0, track=False):
    x_t = torch.randn((100, 4, 17), device=device)
    grads = []
    for t in reversed(range(500)):
        t_b = torch.full((100,), t, device=device, dtype=torch.long)
        if scale > 0:
            x_t.requires_grad_(True)
            sc = oracle(x_t).sum()
            g = torch.autograd.grad(sc, x_t)[0].clamp(-1, 1)
            if track: grads.append(g.norm().item() / 100)
            x_t = x_t.detach()
        else:
            g = 0

        eps = unet(x_t, t_b.float()) - g * scale
        a, acp = 1. - betas[t], alphas_cp[t]
        x_t = (x_t - (1 - a) / torch.sqrt(1 - acp) * eps) / torch.sqrt(a)
        if t > 0: x_t += torch.sqrt(betas[t]) * torch.randn_like(x_t)

    oh = F.one_hot(x_t.argmax(1), 4).permute(0, 2, 1).float()
    return (oh, grads) if track else oh


print("--- Step 4: Generating Comparison Data ---")
base_oh = sample(unet, real_oracle, scale=0.0)
rl_oh, g_flow = sample(unet, real_oracle, scale=1.5, track=True)

with torch.no_grad():
    s_real = real_oracle(torch.from_numpy(all_oh[:100]).to(device)).cpu().numpy().flatten()
    s_base = real_oracle(base_oh).cpu().numpy().flatten()
    s_rl = real_oracle(rl_oh).cpu().numpy().flatten()

# ==============================================================================
# 第5部分: 核心可视化 (全套 7 张图)
# ==============================================================================
print("--- Step 5: Plotting All Results ---")

# 1. 梯度流
plt.figure();
plt.plot(g_flow);
plt.title("Gradient Norm Over Time");
plt.savefig(f"{output_dir}/1_gradient.png")

# 2. 得分分布
df_p = pd.DataFrame(
    {'Method': ['Real'] * 100 + ['Base'] * 100 + ['RL'] * 100, 'Score': np.concatenate([s_real, s_base, s_rl])})
plt.figure();
sns.violinplot(data=df_p, x='Method', y='Score');
plt.savefig(f"{output_dir}/2_violin.png")

# 3. Top-5 Logo
fig, axes = plt.subplots(2, 1, figsize=(10, 5))
for i, (name, oh, sc) in enumerate([('Base', base_oh, s_base), ('RL', rl_oh, s_rl)]):
    top5 = oh[np.argsort(sc)[-5:]].cpu().numpy().mean(0)
    logomaker.Logo(pd.DataFrame(top5.T, columns=list("ACGT")), ax=axes[i])
    axes[i].set_title(name)
plt.tight_layout();
plt.savefig(f"{output_dir}/3_top5.png")

# 4. t-SNE
embs = np.concatenate([real_oracle.get_embedding(torch.from_numpy(all_oh[:100]).to(device)).detach().cpu().numpy(),
                       real_oracle.get_embedding(base_oh).detach().cpu().numpy(),
                       real_oracle.get_embedding(rl_oh).detach().cpu().numpy()])

tsne = TSNE(n_components=2).fit_transform(embs)
plt.figure();
plt.scatter(tsne[:100, 0], tsne[:100, 1], label='Real');
plt.scatter(tsne[100:200, 0], tsne[100:200, 1], label='Base');
plt.scatter(tsne[200:, 0], tsne[200:, 1], label='RL');
plt.legend();
plt.savefig(f"{output_dir}/4_tsne.png")

# 5. 可控性分析
analyze_res = []
for s in [0.0, 0.5, 2.0]:
    oh = sample(unet, real_oracle, scale=s)
    sc = real_oracle(oh).detach().cpu().numpy().flatten()
    analyze_res.append(pd.DataFrame({'Score': sc, 'Guidance': str(s)}))
plt.figure();
sns.boxenplot(data=pd.concat(analyze_res), x='Guidance', y='Score');
plt.savefig(f"{output_dir}/5_control.png")

# 6. 保存文本
with open(f"{output_dir}/generated_sequences.txt", "w") as f:
    strs = one_hot_to_strings(rl_oh)
    for s, v in zip(strs, s_rl): f.write(f"{s},{v:.4f}\n")

print(f"COMPLETE! Files saved in {output_dir}")