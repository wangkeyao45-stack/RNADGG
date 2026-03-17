# -*- coding: utf-8 -*-
import sys
import os
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

# ==============================================================================
# 第0部分: 环境设置和辅助函数
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sns.set_theme(style="whitegrid")
output_dir = "output_plots_multi_objective4_T500"
os.makedirs(output_dir, exist_ok=True)
print(f"所有图表将被保存在 '{output_dir}/' 文件夹中。")

W_R1 = 1.0
W_ENERGY = 1.0


def plot_logo(one_hot_seqs, title=''):
    if isinstance(one_hot_seqs, torch.Tensor):
        one_hot_seqs = one_hot_seqs.detach().cpu().numpy()
    if one_hot_seqs.shape[-1] != 4:
        one_hot_seqs = np.transpose(one_hot_seqs, (0, 2, 1))
    probabilities = np.mean(one_hot_seqs, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9), axis=1)
    information_content = 2.0 - entropy
    heights = probabilities * information_content[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(max(15, one_hot_seqs.shape[1] // 2), 3))
    bases = ['A', 'C', 'G', 'T']
    colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
    for i in range(heights.shape[0]):
        sorted_indices = np.argsort(heights[i])
        y_offset = 0
        for j in sorted_indices:
            base = bases[j]
            height = heights[i, j]
            if height > 0.01:
                ax.text(i + 0.5, y_offset + height / 2, base, ha='center', va='center',
                        fontsize=min(50, int(150 * height)), color=colors[base], weight='bold')
            y_offset += height
    ax.set_xticks(np.arange(heights.shape[0]) + 0.5)
    ax.set_xticklabels(np.arange(1, heights.shape[0] + 1))
    ax.set_xlim(0, heights.shape[0])
    ax.set_ylim(0, 2)
    ax.set_ylabel('Information (bits)')
    ax.set_xlabel('Position')
    ax.set_title(title)
    plt.tight_layout()
    sanitized_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace("/", "")
    plt.savefig(os.path.join(output_dir, f"logo_{sanitized_title}.png"), dpi=300, bbox_inches='tight')
    plt.show()


# ==============================================================================
# 第1部分: PyTorch 模型定义 (完整实现)
# ==============================================================================
SEQUENCE_LENGTH = 50
NUCLEOTIDES = 4
CSV_FILE = "/home/xy_wky/human_5utr_modeling-master/data_pre/f28_utrmatch.csv"


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


class AttentionBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, n_channels)
        self.qkv = nn.Conv1d(n_channels, n_channels * 3, 1)
        self.out = nn.Conv1d(n_channels, n_channels, 1)

    def forward(self, x):
        batch, channel, length = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).view(batch, 3, channel, length)
        q, k, v = qkv.unbind(1)
        attn = torch.einsum('bcl,bck->blk', q, k) * (channel ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('blk,bck->bcl', attn, v)
        return x + self.out(out)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.mlp_time = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.act1(self.norm1(self.conv1(x)))
        h += self.mlp_time(t)[:, :, None]
        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class UpDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, down=True):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.attn = AttentionBlock(out_channels)
        if down:
            self.sample = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.sample = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        x = self.sample(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=64):
        super().__init__()
        time_emb_dim = n_channels * 4
        self.time_embedding = nn.Sequential(TimeEmbedding(n_channels), nn.Linear(n_channels, time_emb_dim), nn.SiLU(),
                                            nn.Linear(time_emb_dim, time_emb_dim))
        self.in_conv = nn.Conv1d(NUCLEOTIDES, n_channels, kernel_size=3, padding=1)
        self.down1 = UpDownBlock(n_channels, n_channels * 2, time_emb_dim, down=True)
        self.down2 = UpDownBlock(n_channels * 2, n_channels * 4, time_emb_dim, down=True)
        self.mid_res1 = ResidualBlock(n_channels * 4, n_channels * 4, time_emb_dim)
        self.mid_attn = AttentionBlock(n_channels * 4)
        self.mid_res2 = ResidualBlock(n_channels * 4, n_channels * 4, time_emb_dim)
        self.up1 = UpDownBlock(n_channels * 8, n_channels * 2, time_emb_dim, down=False)
        self.up2 = UpDownBlock(n_channels * 4, n_channels, time_emb_dim, down=False)
        self.out_res = ResidualBlock(n_channels * 2, n_channels, time_emb_dim)
        self.out_conv = nn.Conv1d(n_channels, NUCLEOTIDES, kernel_size=1)

    def forward(self, x, t):
        t = self.time_embedding(t)
        x = self.in_conv(x)
        x1 = self.down1(x, t)
        x2 = self.down2(x1, t)
        x_mid = self.mid_res1(x2, t)
        x_mid = self.mid_attn(x_mid)
        x_mid = self.mid_res2(x_mid, t)
        x_up = self.up1(torch.cat([x_mid, x2], dim=1), t)
        if x_up.shape[2] != x1.shape[2]: x_up = F.interpolate(x_up, size=x1.shape[2], mode='linear',
                                                              align_corners=False)
        x_up = self.up2(torch.cat([x_up, x1], dim=1), t)
        if x_up.shape[2] != x.shape[2]: x_up = F.interpolate(x_up, size=x.shape[2], mode='linear', align_corners=False)
        x_out = self.out_res(torch.cat([x_up, x], dim=1), t)
        return self.out_conv(x_out)


class Diffusion:
    def __init__(self, noise_predictor_model, T=500):
        self.model = noise_predictor_model
        self.T = T
        self.betas = torch.linspace(1e-4, 0.02, T, device=device)
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
        loss = F.mse_loss(predicted_noise, noise)
        return loss


class OracleCNN(nn.Module):
    def __init__(self):
        super(OracleCNN, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(NUCLEOTIDES, 64, kernel_size=7, padding='same'), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding='same'), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),
        )
        self.fc_net = nn.Sequential(nn.Linear(128 * (SEQUENCE_LENGTH // 4), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, seq): return self.fc_net(self.conv_net(seq))

    def get_embedding(self, seq):
        return self.fc_net[0](self.conv_net(seq))


# ==============================================================================
# 第2部分: 数据加载和预处理 (完整实现)
# ==============================================================================
print("--- Part 2: Loading and Preprocessing Data from CSV ---")
if not os.path.exists(CSV_FILE): sys.exit(f"错误: 数据文件 '{CSV_FILE}' 不存在。")
char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
idx_to_char = {v: k for k, v in char_map.items()}


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
sequences = df['sequence'].tolist()
r1_scores = df['r1'].values.astype(np.float32)
energy_scores = df['energy'].values.astype(np.float32)
processed_data = []
for i in range(len(sequences)):
    encoded_s = one_hot_encode(sequences[i])
    if encoded_s is not None:
        processed_data.append((encoded_s, r1_scores[i], energy_scores[i], sequences[i]))

one_hot_sequences = np.array([item[0] for item in processed_data])
r1_scores = np.array([item[1] for item in processed_data])
energy_scores = np.array([item[2] for item in processed_data])
real_seq_strings = [item[3] for item in processed_data]
print(f"Data loaded and encoded. Final Shape: {one_hot_sequences.shape}")

# ==============================================================================
# 第3部分: 训练双重预言机 (完整实现)
# ==============================================================================
print("\n--- Part 3: Training the Real Oracle Models ---")
# --- 训练 Oracle for r1 ---
print("--- Training Oracle for r1 score ---")
X_train, X_val, y_train, y_val = train_test_split(one_hot_sequences, r1_scores, test_size=0.2, random_state=42)
oracle_r1_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32,
                                    shuffle=True)
oracle_r1_val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=32)
real_oracle_r1 = OracleCNN().to(device)
oracle_r1_optimizer = optim.Adam(real_oracle_r1.parameters(), lr=1e-3)
criterion = nn.MSELoss()
ORACLE_EPOCHS = 20
for epoch in range(ORACLE_EPOCHS):
    real_oracle_r1.train()
    train_loss_epoch = 0
    for seqs, labels in oracle_r1_train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        oracle_r1_optimizer.zero_grad()
        outputs = real_oracle_r1(seqs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        oracle_r1_optimizer.step()
        train_loss_epoch += loss.item()
    real_oracle_r1.eval()
    val_loss = 0
    with torch.no_grad():
        for seqs, labels in oracle_r1_val_loader:
            outputs = real_oracle_r1(seqs.to(device)).squeeze()
            val_loss += criterion(outputs, labels.to(device)).item()
    if (epoch + 1) % 5 == 0:
        print(
            f"Oracle Epoch [{epoch + 1}/{ORACLE_EPOCHS}], Train Loss: {train_loss_epoch / len(oracle_r1_train_loader):.4f}, Val Loss: {val_loss / len(oracle_r1_val_loader):.4f}")
print("--- Oracle for r1 training finished. ---")

# --- 训练 Oracle for energy ---
print("\n--- Training Oracle for energy score ---")
X_train_e, X_val_e, y_train_e, y_val_e = train_test_split(one_hot_sequences, energy_scores, test_size=0.2,
                                                          random_state=42)
oracle_energy_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_e), torch.from_numpy(y_train_e)),
                                        batch_size=32, shuffle=True)
oracle_energy_val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_e), torch.from_numpy(y_val_e)),
                                      batch_size=32)
real_oracle_energy = OracleCNN().to(device)
oracle_energy_optimizer = optim.Adam(real_oracle_energy.parameters(), lr=1e-3)
for epoch in range(ORACLE_EPOCHS):
    real_oracle_energy.train()
    train_loss_epoch = 0
    for seqs, labels in oracle_energy_train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        oracle_energy_optimizer.zero_grad()
        outputs = real_oracle_energy(seqs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        oracle_energy_optimizer.step()
        train_loss_epoch += loss.item()
    real_oracle_energy.eval()
    val_loss = 0
    with torch.no_grad():
        for seqs, labels in oracle_energy_val_loader:
            outputs = real_oracle_energy(seqs.to(device)).squeeze()
            val_loss += criterion(outputs, labels.to(device)).item()
    if (epoch + 1) % 5 == 0:
        print(
            f"Oracle Epoch [{epoch + 1}/{ORACLE_EPOCHS}], Train Loss: {train_loss_epoch / len(oracle_energy_train_loader):.4f}, Val Loss: {val_loss / len(oracle_energy_val_loader):.4f}")
print("--- Oracle for energy training finished. ---")
torch.cuda.empty_cache()

r1_mean, r1_std = torch.tensor(r1_scores.mean(), device=device), torch.tensor(r1_scores.std(), device=device)
energy_mean, energy_std = torch.tensor(energy_scores.mean(), device=device), torch.tensor(energy_scores.std(),
                                                                                          device=device)


def multi_objective_reward_oracle(sequences_n_c_l):
    pred_r1 = real_oracle_r1(sequences_n_c_l).squeeze()
    pred_energy = real_oracle_energy(sequences_n_c_l).squeeze()

    norm_r1 = (pred_r1 - r1_mean) / r1_std
    norm_energy = (pred_energy - energy_mean) / energy_std

    return W_R1 * norm_r1 - W_ENERGY * norm_energy


# ==============================================================================
# 第4部分: 训练扩散模型
# ==============================================================================
print("\n--- Part 4: Training the Diffusion Model ---")
diffusion_train_loader = DataLoader(TensorDataset(torch.from_numpy(one_hot_sequences)), batch_size=64, shuffle=True)
unet = UNet().to(device)
diffusion = Diffusion(unet)
diffusion_optimizer = optim.Adam(unet.parameters(), lr=1e-4)
DIFFUSION_EPOCHS = 20
for epoch in range(DIFFUSION_EPOCHS):
    pbar = tqdm(diffusion_train_loader, desc=f"Diffusion Epoch {epoch + 1}/{DIFFUSION_EPOCHS}")
    for i, (real_seqs,) in enumerate(pbar):
        diffusion_optimizer.zero_grad()
        loss = diffusion.train_step(real_seqs.to(device))
        loss.backward()
        diffusion_optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
print("--- Diffusion Model training finished. ---")
unet.eval()
torch.cuda.empty_cache()

# ==============================================================================
# 第5部分: 强化学习引导式生成
# ==============================================================================
print("\n--- Part 5: Multi-Objective RL-Guided Generative Process ---")


def guided_sampling(diffusion_model, oracle_model, batch_size, guidance_scale=0.01):
    x_t = torch.randn((batch_size, NUCLEOTIDES, SEQUENCE_LENGTH), device=device)
    for t in tqdm(reversed(range(diffusion_model.T)), desc="Guided Sampling", total=diffusion_model.T):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        with torch.enable_grad():
            x_t.requires_grad_()
            scores = oracle_model(x_t).sum()
            grad = torch.autograd.grad(scores, x_t)[0].clamp(-1, 1)
        predicted_noise = diffusion_model.model(x_t.detach(), t_tensor.float())
        alpha_t = diffusion_model.alphas[t]
        alpha_t_cumprod = diffusion_model.alphas_cumprod[t]
        guided_noise = predicted_noise - torch.sqrt(1. - alpha_t_cumprod) * grad * guidance_scale
        x_t_minus_1_mean = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod) * guided_noise) / torch.sqrt(alpha_t)
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = x_t_minus_1_mean + torch.sqrt(diffusion_model.betas[t]) * noise
        else:
            x_t = x_t_minus_1_mean
    final_indices = torch.argmax(x_t, dim=1)
    return F.one_hot(final_indices, num_classes=NUCLEOTIDES).float().permute(0, 2, 1)


# 运行引导式生成
SAMPLING_BATCH_SIZE = 16
GUIDANCE_STRENGTH = 0.1
rl_diffusion_seqs = guided_sampling(diffusion, multi_objective_reward_oracle, batch_size=SAMPLING_BATCH_SIZE,
                                    guidance_scale=GUIDANCE_STRENGTH)
baseline_diffusion_seqs = guided_sampling(diffusion, multi_objective_reward_oracle, batch_size=SAMPLING_BATCH_SIZE,
                                          guidance_scale=0.0)

# ==============================================================================
# 第6部分: 评估与可视化
# ==============================================================================
print("\n--- Part 6: Final Evaluation and Visualization ---")
with torch.no_grad():
    base_pred_r1 = real_oracle_r1(baseline_diffusion_seqs.to(device)).cpu().numpy().flatten()
    base_pred_energy = real_oracle_energy(baseline_diffusion_seqs.to(device)).cpu().numpy().flatten()
    base_composite_rewards = multi_objective_reward_oracle(baseline_diffusion_seqs.to(device)).cpu().numpy().flatten()

    rl_pred_r1 = real_oracle_r1(rl_diffusion_seqs.to(device)).cpu().numpy().flatten()
    rl_pred_energy = real_oracle_energy(rl_diffusion_seqs.to(device)).cpu().numpy().flatten()
    rl_composite_rewards = multi_objective_reward_oracle(rl_diffusion_seqs.to(device)).cpu().numpy().flatten()

print("--- Generating Reward Distribution Plots ---")
df_r1 = pd.DataFrame({'Model': ['Baseline'] * len(base_pred_r1) + ['RL-Guided'] * len(rl_pred_r1),
                      'Predicted r1 Score': np.concatenate([base_pred_r1, rl_pred_r1])})
plt.figure(figsize=(8, 6))
sns.violinplot(x='Model', y='Predicted r1 Score', data=df_r1)
plt.title("Distribution of Predicted r1 Scores")
plt.savefig(os.path.join(output_dir, "reward_dist_r1.png"), dpi=300)
plt.show()

df_energy = pd.DataFrame({'Model': ['Baseline'] * len(base_pred_energy) + ['RL-Guided'] * len(rl_pred_energy),
                          'Predicted Energy': np.concatenate([base_pred_energy, rl_pred_energy])})
plt.figure(figsize=(8, 6))
sns.violinplot(x='Model', y='Predicted Energy', data=df_energy)
plt.title("Distribution of Predicted Energy Scores (Lower is Better)")
plt.savefig(os.path.join(output_dir, "reward_dist_energy.png"), dpi=300)
plt.show()

# ==============================================================================
# 第7部分: 综合评估模块
# ==============================================================================
print("\n--- Part 7: Comprehensive Sequence Evaluation ---")


def one_hot_to_strings(one_hot_tensor):
    indices = torch.argmax(one_hot_tensor.permute(0, 2, 1), dim=2)
    return ["".join([idx_to_char.get(i.item(), 'N') for i in seq]) for seq in indices]


N_EVAL_SAMPLES = 200
real_seqs_eval = real_seq_strings[:N_EVAL_SAMPLES]
baseline_seqs_str = one_hot_to_strings(baseline_diffusion_seqs)[:N_EVAL_SAMPLES]
rl_seqs_str = one_hot_to_strings(rl_diffusion_seqs)[:N_EVAL_SAMPLES]

print("\n--- [Metric 1] Calculating Levenshtein Distance ---")


def calculate_avg_min_levenshtein(generated_seqs, real_seqs):
    total_min_dist = 0
    if not generated_seqs: return float('inf')
    for gen_seq in tqdm(generated_seqs, desc="Levenshtein"):
        min_dist = min([Levenshtein.distance(gen_seq, real_seq) for real_seq in real_seqs])
        total_min_dist += min_dist
    return total_min_dist / len(generated_seqs)


dist_baseline = calculate_avg_min_levenshtein(baseline_seqs_str, real_seqs_eval)
dist_rl = calculate_avg_min_levenshtein(rl_seqs_str, real_seqs_eval)
print(f"  Average Min Levenshtein Distance (Baseline vs Real): {dist_baseline:.4f} (Lower is better)")
print(f"  Average Min Levenshtein Distance (RL-Guided vs Real): {dist_rl:.4f} (Lower is better)")

print("\n--- [Metric 2] Calculating 4-mer JSD ---")


def get_kmer_dist(sequences, k=4):
    all_kmers = [seq[i:i + k] for seq in sequences for i in range(len(seq) - k + 1)]
    counts = Counter(all_kmers)
    total = sum(counts.values())
    return {kmer: v / total for kmer, v in counts.items()} if total > 0 else {}


real_kmer_dist = get_kmer_dist(real_seqs_eval)
baseline_kmer_dist = get_kmer_dist(baseline_seqs_str)
rl_kmer_dist = get_kmer_dist(rl_seqs_str)
all_kmers_vocab = sorted(list(set(real_kmer_dist.keys()) | set(baseline_kmer_dist.keys()) | set(rl_kmer_dist.keys())))
p_real = np.array([real_kmer_dist.get(k, 0) for k in all_kmers_vocab])
p_baseline = np.array([baseline_kmer_dist.get(k, 0) for k in all_kmers_vocab])
p_rl = np.array([rl_kmer_dist.get(k, 0) for k in all_kmers_vocab])
jsd_baseline = jensenshannon(p_real, p_baseline, base=2.0) if len(p_real) > 0 and len(p_baseline) > 0 else float('inf')
jsd_rl = jensenshannon(p_real, p_rl, base=2.0) if len(p_real) > 0 and len(p_rl) > 0 else float('inf')
print(f"  4-mer Jenson-Shannon Divergence (Baseline vs Real): {jsd_baseline:.4f} (Lower is better)")
print(f"  4-mer Jenson-Shannon Divergence (RL-Guided vs Real): {jsd_rl:.4f} (Lower is better)")

print("\n--- [Metric 3] Calculating GC Content Distribution ---")


def get_gc_contents(sequences):
    return [(s.count('G') + s.count('C')) / len(s) * 100 for s in sequences if len(s) > 0]


gc_real = get_gc_contents(real_seqs_eval)
gc_baseline = get_gc_contents(baseline_seqs_str)
gc_rl = get_gc_contents(rl_seqs_str)
plt.figure(figsize=(10, 6))
sns.kdeplot(gc_real, label=f'Real (Avg: {np.mean(gc_real):.2f}%)', fill=True, clip=(0, 100))
sns.kdeplot(gc_baseline, label=f'Baseline (Avg: {np.mean(gc_baseline):.2f}%)', fill=True, clip=(0, 100))
sns.kdeplot(gc_rl, label=f'RL-Guided (Avg: {np.mean(gc_rl):.2f}%)', fill=True, clip=(0, 100))
plt.title('GC Content Distribution Comparison')
plt.xlabel('GC Content (%)')
plt.legend()
plt.savefig(os.path.join(output_dir, "gc_content_distribution.png"), dpi=300)
plt.show()

print("\n--- [Metric 4] Performing t-SNE Visualization ---")


def get_kmer_features(sequences, k=4):
    all_kmers_in_real = sorted(list(get_kmer_dist(sequences, k=k).keys()))
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers_in_real)}
    features = np.zeros((len(sequences), len(kmer_to_idx)))
    for i, seq in enumerate(sequences):
        dist = get_kmer_dist([seq], k=k)
        for kmer, freq in dist.items():
            if kmer in kmer_to_idx:
                features[i, kmer_to_idx[kmer]] = freq
    return features


all_seqs_for_tsne = real_seqs_eval + baseline_seqs_str + rl_seqs_str
labels = ['Real'] * len(real_seqs_eval) + ['Baseline'] * len(baseline_seqs_str) + ['RL-Guided'] * len(rl_seqs_str)
kmer_features = get_kmer_features(all_seqs_for_tsne, k=3)
tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(kmer_features)
df_tsne = pd.DataFrame({'x': tsne_results[:, 0], 'y': tsne_results[:, 1], 'label': labels})
plt.figure(figsize=(10, 8))
sns.scatterplot(x="x", y="y", hue="label", palette=sns.color_palette("hls", 3), data=df_tsne, s=50, alpha=0.7)
plt.title('t-SNE Visualization of Sequence Distributions (3-mer space)')
plt.legend()
plt.savefig(os.path.join(output_dir, "tsne_visualization.png"), dpi=300)
plt.show()

print("\n--- [Metric 5] Calculating Fréchet Inception Distance (FID) ---")


@torch.no_grad()
def get_embeddings(one_hot_seqs, oracle_model):
    oracle_model.eval()
    embeddings = oracle_model.get_embedding(one_hot_seqs.to(device))
    return embeddings.cpu().numpy()


def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


real_embeddings = get_embeddings(torch.from_numpy(one_hot_sequences[:N_EVAL_SAMPLES]), real_oracle_r1)
baseline_embeddings = get_embeddings(baseline_diffusion_seqs, real_oracle_r1)
rl_embeddings = get_embeddings(rl_diffusion_seqs, real_oracle_r1)
fid_baseline = calculate_fid(real_embeddings, baseline_embeddings)
fid_rl = calculate_fid(real_embeddings, rl_embeddings)
print(f"  Fréchet Inception Distance (Baseline vs Real): {fid_baseline:.4f} (Lower is better)")
print(f"  Fréchet Inception Distance (RL-Guided vs Real): {fid_rl:.4f} (Lower is better)")

# ==============================================================================
# 第8部分: 最终量化指标
# ==============================================================================
print("\n--- Part 8: Final Quantitative Metrics ---")
print(f"  --- Objective: r1 Score (Higher is Better) ---")
print(f"  Average Predicted r1 (Baseline): {np.mean(base_pred_r1):.4f}")
print(f"  Average Predicted r1 (RL-Guided): {np.mean(rl_pred_r1):.4f}")
print(f"\n  --- Objective: Energy Score (Lower is Better) ---")
print(f"  Average Predicted Energy (Baseline): {np.mean(base_pred_energy):.4f}")
print(f"  Average Predicted Energy (RL-Guided): {np.mean(rl_pred_energy):.4f}")
print(f"\n  --- Composite Score (Weighted & Normalized) ---")
print(f"  Average Composite Reward (Baseline): {np.mean(base_composite_rewards):.4f}")
print(f"  Average Composite Reward (RL-Guided): {np.mean(rl_composite_rewards):.4f}")
best_rl_idx = np.argmax(rl_composite_rewards)
best_composite_score = rl_composite_rewards[best_rl_idx]
best_r1_score = rl_pred_r1[best_rl_idx]
best_energy_score = rl_pred_energy[best_rl_idx]
best_seq_indices = torch.argmax(rl_diffusion_seqs[best_rl_idx], dim=0).cpu().numpy()
best_seq_str = "".join([idx_to_char.get(i, 'N') for i in best_seq_indices])
print(f"\n--- Best Sequence Generated by Multi-Objective RL ---")
print(f"  Sequence: {best_seq_str}")
print(f"  Composite Reward: {best_composite_score:.4f}")
print(f"  Predicted r1 Score: {best_r1_score:.4f}")
print(f"  Predicted Energy: {best_energy_score:.4f}")