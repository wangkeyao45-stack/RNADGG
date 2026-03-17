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
from sklearn.manifold import TSNE
from tqdm import tqdm
import math
import Levenshtein
from scipy.spatial.distance import jensenshannon
from scipy.linalg import sqrtm
from collections import Counter
import logomaker


# ==============================================================================
# 第0部分: 环境配置与全局参数
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

# 输出目录
output_dir = "output_master_combined"
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved to: {output_dir}/")

# 数据参数
SEQUENCE_LENGTH = 59
NUCLEOTIDES = 4
CSV_FILE = "toehold_data.csv"


# ==============================================================================
# 第1部分: 模型定义 (UNet 修复版 + Cosine Diffusion)
# ==============================================================================
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
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_c))
        self.conv1 = nn.Conv1d(in_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.short = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t):
        h = F.silu(self.norm1(self.conv1(x))) + self.mlp(t)[:, :, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.short(x)


class AttentionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.qkv = nn.Conv1d(c, c * 3, 1)
        self.out = nn.Conv1d(c, c, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.qkv(self.norm(x)).view(b, 3, c, l)
        q, k, v = qkv.unbind(1)
        attn = (torch.einsum('bcl,bck->blk', q, k) * (c ** -0.5)).softmax(-1)
        return x + self.out(torch.einsum('blk,bck->bcl', attn, v))


# --- [关键修复] UNet 类 ---
class UNet(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        t_dim = c * 4
        self.time_emb = nn.Sequential(TimeEmbedding(c), nn.Linear(c, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.head = nn.Conv1d(4, c, 3, padding=1)

        # 下采样
        self.down1 = nn.Sequential(ResidualBlock(c, c * 2, t_dim), AttentionBlock(c * 2),
                                   nn.Conv1d(c * 2, c * 2, 3, 2, 1))

        # 中间层
        self.mid = nn.Sequential(ResidualBlock(c * 2, c * 2, t_dim), AttentionBlock(c * 2),
                                 ResidualBlock(c * 2, c * 2, t_dim))

        # 上采样
        self.up1 = nn.Sequential(ResidualBlock(c * 4, c, t_dim), AttentionBlock(c), nn.ConvTranspose1d(c, c, 4, 2, 1))

        # 输出层 (拆分为 ResidualBlock 和 Conv1d)
        self.out_res = ResidualBlock(c * 2, c, t_dim)
        self.out_conv = nn.Conv1d(c, 4, 1)

    def forward(self, x, t):
        t = self.time_emb(t)
        x_in = self.head(x)

        d1 = self.down1[0](x_in, t);
        d1 = self.down1[1](d1);
        d1_pooled = self.down1[2](d1)

        mid = self.mid[0](d1_pooled, t);
        mid = self.mid[1](mid);
        mid = self.mid[2](mid, t)

        # Skip connection concat
        u1 = self.up1[0](torch.cat([mid, d1_pooled], 1), t);
        u1 = self.up1[1](u1);
        u1 = self.up1[2](u1)

        if u1.shape[2] != x_in.shape[2]: u1 = F.interpolate(u1, size=x_in.shape[2])

        # [关键修复] 显式调用输出层
        out = self.out_res(torch.cat([u1, x_in], 1), t)
        return self.out_conv(out)  # 确保输出维度是 [Batch, 4, Length]


class Diffusion:
    def __init__(self, model, T=1000):
        self.model = model
        self.T = T
        # Cosine Schedule
        s = 0.008
        steps = T + 1
        x = torch.linspace(0, T, steps, device=device)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(betas, 0.0001, 0.9999)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def train_step(self, x_0):
        t = torch.randint(0, self.T, (x_0.shape[0],), device=device)
        noise = torch.randn_like(x_0)
        mean = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1) * x_0
        std = torch.sqrt(1. - self.alphas_cumprod[t]).view(-1, 1, 1)
        x_t = mean + std * noise
        pred_noise = self.model(x_t, t.float())
        return F.mse_loss(pred_noise, noise)


class OracleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding='same'), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding='same'), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(nn.Linear(128 * (SEQUENCE_LENGTH // 4), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x): return self.fc(self.conv(x))

    def get_embedding(self, x): return self.fc[0](self.conv(x))


# ==============================================================================
# 第2部分: 数据加载
# ==============================================================================
print("--- Part 2: Data Loading ---")
char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
idx_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


def one_hot_encode(seq):
    if len(seq) != SEQUENCE_LENGTH: return None
    enc = np.zeros((4, SEQUENCE_LENGTH), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        if c in char_map: enc[char_map[c], i] = 1.0
    return enc


if not os.path.exists(CSV_FILE):
    print("WARNING: CSV not found. Generating Dummy Data.")
    sequences = ["".join(random.choices(['A', 'C', 'G', 'T'], k=SEQUENCE_LENGTH)) for _ in range(5000)]
    scores = np.random.rand(5000).astype(np.float32)
else:
    df = pd.read_csv(CSV_FILE)
    sequences = df['full_sequence'].tolist()
    scores = df['ON'].values.astype(np.float32)

data = [(one_hot_encode(s), sc) for s, sc in zip(sequences, scores) if one_hot_encode(s) is not None]
one_hot_all = np.array([d[0] for d in data])
scores_all = np.array([d[1] for d in data])
print(f"Loaded {len(one_hot_all)} sequences.")

# ==============================================================================
# 第3部分: 训练 Oracle
# ==============================================================================
print("--- Part 3: Training Robust Oracle ---")
X_train, X_val, y_train, y_val = train_test_split(one_hot_all, scores_all, test_size=0.2, random_state=42)
oracle_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32,
                           shuffle=True)
oracle = OracleCNN().to(device)
opt_oracle = optim.Adam(oracle.parameters(), lr=1e-3)

for epoch in range(20):
    oracle.train()
    loss_sum = 0
    for x, y in oracle_loader:
        x, y = x.to(device), y.to(device)
        opt_oracle.zero_grad()
        noise = torch.randn_like(x) * 0.1
        loss = F.mse_loss(oracle(x + noise).squeeze(), y)
        loss.backward()
        opt_oracle.step()
        loss_sum += loss.item()
    if (epoch + 1) % 5 == 0: print(f"Oracle Epoch {epoch + 1}, Loss: {loss_sum / len(oracle_loader):.4f}")
oracle.eval()

# ==============================================================================
# 第4部分: 训练 Diffusion
# ==============================================================================
print("--- Part 4: Training Diffusion ---")
diff_loader = DataLoader(TensorDataset(torch.from_numpy(one_hot_all)), batch_size=64, shuffle=True)
unet = UNet().to(device)
diffusion = Diffusion(unet, T=1000)
opt_diff = optim.Adam(unet.parameters(), lr=1e-4)

for epoch in range(20):
    unet.train()
    pbar = tqdm(diff_loader, desc=f"Diff Epoch {epoch + 1}", leave=False)
    for (x,) in pbar:
        opt_diff.zero_grad()
        loss = diffusion.train_step(x.to(device))
        loss.backward()
        opt_diff.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")


# ==============================================================================
# 第5部分: 采样与遗传算法
# ==============================================================================
def guided_sampling(n_samples, guidance_scale=1.0, target_val=None, track_grad=False):
    unet.eval()
    x = torch.randn((n_samples, 4, SEQUENCE_LENGTH), device=device)
    grad_logs = []

    for t in tqdm(reversed(range(diffusion.T)), desc="Sampling", total=diffusion.T, leave=False):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

        grad = torch.zeros_like(x)
        if guidance_scale > 0:
            with torch.enable_grad():
                x_in = x.detach().requires_grad_()
                pred = oracle(x_in).squeeze()

                if target_val is not None:
                    tgt = torch.full_like(pred, target_val)
                    obj = -((pred - tgt) ** 2).sum()
                else:
                    obj = pred.sum()

                g = torch.autograd.grad(obj, x_in)[0]
                if track_grad: grad_logs.append(g.norm().item() / n_samples)
                grad = torch.clamp(g, -0.1, 0.1)

        with torch.no_grad():
            pred_noise = unet(x, t_batch.float())
            alpha_t = diffusion.alphas[t]
            alpha_cum = diffusion.alphas_cumprod[t]
            beta_t = diffusion.betas[t]

            guided_noise = pred_noise - torch.sqrt(1 - alpha_cum) * grad * guidance_scale

            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1 - alpha_cum)
            mean = coef1 * (x - coef2 * guided_noise)

            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

    final_seq = F.one_hot(x.argmax(1), 4).float().permute(0, 2, 1)
    if track_grad: return final_seq, grad_logs
    return final_seq


class GeneticAlgorithm:
    def __init__(self, oracle, pop_size=100):
        self.oracle = oracle
        self.pop_size = pop_size

    def run(self, generations=50):
        pop = ["".join(random.choices(['A', 'C', 'G', 'T'], k=SEQUENCE_LENGTH)) for _ in range(self.pop_size)]

        for _ in tqdm(range(generations), desc="GA Evol", leave=False):
            batch = np.zeros((len(pop), 4, SEQUENCE_LENGTH), dtype=np.float32)
            for i, s in enumerate(pop):
                for j, c in enumerate(s): batch[i, char_map[c], j] = 1.0

            with torch.no_grad():
                scores = self.oracle(torch.tensor(batch).to(device)).cpu().numpy().flatten()

            survivors = [pop[i] for i in np.argsort(scores)[::-1][:self.pop_size // 2]]

            next_gen = survivors[:]
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(survivors, 2)
                pt = random.randint(1, SEQUENCE_LENGTH - 1)
                child = p1[:pt] + p2[pt:]
                if random.random() < 0.1:
                    mp = random.randint(0, SEQUENCE_LENGTH - 1)
                    child = child[:mp] + random.choice(['A', 'C', 'G', 'T']) + child[mp + 1:]
                next_gen.append(child)
            pop = next_gen

        batch = np.zeros((len(pop), 4, SEQUENCE_LENGTH), dtype=np.float32)
        for i, s in enumerate(pop):
            for j, c in enumerate(s): batch[i, char_map[c], j] = 1.0
        return torch.tensor(batch).to(device)


# ==============================================================================
# 第6部分: 生成与基准测试
# ==============================================================================
print("--- Part 6: Generating & Benchmarking ---")
N_EVAL = 200

# 1. Real
idx = np.random.choice(len(one_hot_all), N_EVAL, replace=False)
real_oh = torch.from_numpy(one_hot_all[idx]).to(device)

# 2. Random
rand_str = ["".join(random.choices(['A', 'C', 'G', 'T'], k=SEQUENCE_LENGTH)) for _ in range(N_EVAL)]
rand_oh = torch.zeros(N_EVAL, 4, SEQUENCE_LENGTH).to(device)
for i, s in enumerate(rand_str):
    for j, c in enumerate(s): rand_oh[i, char_map[c], j] = 1.0

# 3. GA
ga = GeneticAlgorithm(oracle, pop_size=N_EVAL)
ga_oh = ga.run(generations=50)

# 4. Ours (RNADGG)
ours_oh, grad_logs = guided_sampling(N_EVAL, guidance_scale=2.0, track_grad=True)


# 计算指标
def get_scores(oh):
    with torch.no_grad(): return oracle(oh).cpu().numpy().flatten()


s_real = get_scores(real_oh)
s_rand = get_scores(rand_oh)
s_ga = get_scores(ga_oh)
s_ours = get_scores(ours_oh)


def to_str(oh):
    idx = oh.argmax(1).cpu().numpy()
    return ["".join([idx_to_char[i] for i in s]) for s in idx]


def calc_metrics(gen_oh, real_oh):
    gen_str, real_str = to_str(gen_oh), to_str(real_oh)
    div = 0
    for _ in range(1000):
        a, b = random.sample(gen_str, 2)
        div += Levenshtein.distance(a, b)
    div /= 1000

    with torch.no_grad():
        emb_gen = oracle.get_embedding(gen_oh).cpu().numpy()
        emb_real = oracle.get_embedding(real_oh).cpu().numpy()
    mu1, sig1 = np.mean(emb_gen, 0), np.cov(emb_gen, rowvar=False)
    mu2, sig2 = np.mean(emb_real, 0), np.cov(emb_real, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sig1.dot(sig2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sig1 + sig2 - 2 * covmean)
    return div, fid


print("Calculating Metrics...")
div_real, _ = calc_metrics(real_oh, real_oh)
div_ga, fid_ga = calc_metrics(ga_oh, real_oh)
div_ours, fid_ours = calc_metrics(ours_oh, real_oh)

print(f"GA   -> Score: {s_ga.mean():.3f}, Diversity: {div_ga:.2f}, FID: {fid_ga:.2f}")
print(f"Ours -> Score: {s_ours.mean():.3f}, Diversity: {div_ours:.2f}, FID: {fid_ours:.2f}")

# ==============================================================================
# 第7部分: 可视化
# ==============================================================================
print("--- Part 7: Visualizing ---")

# 7.1 分数分布
df_viol = pd.DataFrame({
    'Score': np.concatenate([s_real, s_rand, s_ga, s_ours]),
    'Method': ['Real'] * N_EVAL + ['Random'] * N_EVAL + ['GA'] * N_EVAL + ['Ours'] * N_EVAL
})
plt.figure(figsize=(8, 6))
sns.violinplot(x='Method', y='Score', data=df_viol, palette="muted")
plt.title("Score Distribution Comparison")
plt.savefig(f"{output_dir}/1_score_violin.png")
plt.show()

# 7.2 Pareto Scatter
methods = ['Real', 'GA', 'Ours']
divs = [div_real, div_ga, div_ours]
means = [s_real.mean(), s_ga.mean(), s_ours.mean()]
plt.figure(figsize=(7, 5))
for i, m in enumerate(methods):
    plt.scatter(divs[i], means[i], s=200, label=m, alpha=0.8)
    plt.text(divs[i], means[i] + 0.02, m, ha='center', weight='bold')
plt.xlabel("Diversity (Levenshtein)");
plt.ylabel("Mean Score")
plt.title("Diversity vs Functionality Trade-off")
plt.grid(True, linestyle='--')
plt.savefig(f"{output_dir}/2_pareto_scatter.png")
plt.show()

# 7.3 梯度动态
plt.figure(figsize=(8, 4))
plt.plot(grad_logs, color='purple')
plt.xlabel("Denoising Steps (T -> 0)");
plt.ylabel("Grad Norm")
plt.title("RL Guidance Gradient Dynamics")
plt.savefig(f"{output_dir}/3_grad_dynamics.png")
plt.show()

# 7.4 t-SNE
print("Running t-SNE...")
with torch.no_grad():
    emb_real = oracle.get_embedding(real_oh).cpu().numpy()
    emb_ga = oracle.get_embedding(ga_oh).cpu().numpy()
    emb_ours = oracle.get_embedding(ours_oh).cpu().numpy()
all_embs = np.concatenate([emb_real, emb_ga, emb_ours], 0)
labels = ['Real'] * N_EVAL + ['GA'] * N_EVAL + ['Ours'] * N_EVAL
tsne = TSNE(n_components=2, perplexity=30).fit_transform(all_embs)
df_tsne = pd.DataFrame({'D1': tsne[:, 0], 'D2': tsne[:, 1], 'Type': labels})
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_tsne, x='D1', y='D2', hue='Type', style='Type', s=60, alpha=0.7)
plt.title("Latent Space Visualization (t-SNE)")
plt.savefig(f"{output_dir}/4_tsne.png")
plt.show()

# 7.5 定值生成验证
targets = [np.percentile(scores_all, 50), np.percentile(scores_all, 90)]
plt.figure(figsize=(8, 5))
for t in targets:
    gen = guided_sampling(50, guidance_scale=2.0, target_val=t)
    sc = get_scores(gen)
    sns.kdeplot(sc, label=f"Target {t:.2f}", fill=True)
    plt.axvline(t, color='red', linestyle='--')
plt.title("Conditional Generation Verification")
plt.legend()
plt.savefig(f"{output_dir}/5_target_verify.png")
plt.show()


# 7.6 Sequence Logos
def plot_logo(oh, title):
    pwm = pd.DataFrame(oh.mean(0).cpu().numpy().T, columns=['A', 'C', 'G', 'T'])
    fig, ax = plt.subplots(figsize=(10, 3))
    logomaker.Logo(pwm, ax=ax)
    ax.set_title(title)
    plt.savefig(f"{output_dir}/6_logo_{title}.png")
    plt.show()


top_idx = np.argsort(s_ours)[-20:]
plot_logo(ours_oh[top_idx.copy()], "Top_20_Generated")

print("\nAll tasks completed. Master script finished.")