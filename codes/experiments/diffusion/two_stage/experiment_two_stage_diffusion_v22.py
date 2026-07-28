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
# --- [修改点1] 引入 logomaker ---
import logomaker

# ==============================================================================
# 第0部分: 环境设置和辅助函数
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sns.set_theme(style="whitegrid")
output_dir = "output_plots_toehold_final22"
os.makedirs(output_dir, exist_ok=True)
print(f"所有图表将被保存在 '{output_dir}/' 文件夹中。")

W_on = 1.0
W_off = 1.0

# --- [修改点2] 恢复使用 logomaker 的 plot_logo 函数 ---
def plot_logo(one_hot_seqs, title=''):
    """
    使用logomaker库绘制专业的序列标识图 (恢复第一版风格)
    """
    if isinstance(one_hot_seqs, torch.Tensor):
        one_hot_seqs = one_hot_seqs.detach().cpu().numpy()

    # 确保形状是 (Length, 4)
    if one_hot_seqs.shape[-1] != 4:
        one_hot_seqs = np.transpose(one_hot_seqs, (0, 2, 1))

    # 计算 PWM (Position Weight Matrix)
    pwm = np.mean(one_hot_seqs, axis=0)
    pwm_df = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])

    seq_length = pwm_df.shape[0]
    # 动态调整图片宽度
    fig_width = max(10, seq_length * 0.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 2.5))

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 使用 logomaker 绘制
    logo = logomaker.Logo(pwm_df,
                          ax=ax,
                          shade_below=0.5,
                          fade_below=0.5,
                          font_name='sans-serif',
                          color_scheme='classic') # 或者 'colorblind_safe'

    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.ax.set_ylabel("Information (bits)", labelpad=2, fontsize=10)
    logo.ax.set_xlabel("Position", fontsize=10)
    logo.ax.set_xticks(range(seq_length))
    logo.ax.set_xticklabels(range(1, seq_length + 1))
    logo.ax.spines['left'].set_linewidth(1.5)
    logo.ax.spines['bottom'].set_linewidth(1.5)
    logo.ax.grid(False)

    if title:
        logo.ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()
    sanitized_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace("/", "")
    if not sanitized_title: sanitized_title = "sequence_logo"

    plt.savefig(os.path.join(output_dir, f"logo_{sanitized_title}.png"), dpi=300, bbox_inches='tight')
    plt.show()


# ==============================================================================
# 第1部分: PyTorch 模型定义
# ==============================================================================
SEQUENCE_LENGTH = 59
NUCLEOTIDES = 4
CSV_FILE = "toehold_data.csv"


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
# 第2部分: 数据加载和预处理
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
sequences = df['full_sequence'].tolist()
on_scores = df['ON'].values.astype(np.float32)
off_scores = df['OFF'].values.astype(np.float32)
processed_data = []
for i in range(len(sequences)):
    encoded_s = one_hot_encode(sequences[i])
    if encoded_s is not None:
        processed_data.append((encoded_s, on_scores[i], off_scores[i], sequences[i]))

one_hot_sequences = np.array([item[0] for item in processed_data])
on_scores = np.array([item[1] for item in processed_data])
off_scores = np.array([item[2] for item in processed_data])
real_seq_strings = [item[3] for item in processed_data]
print(f"Data loaded and encoded. Final Shape: {one_hot_sequences.shape}")

# ==============================================================================
# 第3部分: 训练双重预言机
# ==============================================================================
print("\n--- Part 3: Training the Real Oracle Models ---")
# --- 训练 Oracle for on ---
print("--- Training Oracle for ON score ---")
X_train, X_val, y_train, y_val = train_test_split(one_hot_sequences, on_scores, test_size=0.2, random_state=42)
oracle_on_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32,
                                    shuffle=True)
oracle_on_val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=32)
real_oracle_on = OracleCNN().to(device)
oracle_on_optimizer = optim.Adam(real_oracle_on.parameters(), lr=1e-3)
criterion = nn.MSELoss()
ORACLE_EPOCHS = 20
for epoch in range(ORACLE_EPOCHS):
    real_oracle_on.train()
    train_loss_epoch = 0
    for seqs, labels in oracle_on_train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        oracle_on_optimizer.zero_grad()
        outputs = real_oracle_on(seqs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        oracle_on_optimizer.step()
        train_loss_epoch += loss.item()
    real_oracle_on.eval()
    val_loss = 0
    with torch.no_grad():
        for seqs, labels in oracle_on_val_loader:
            outputs = real_oracle_on(seqs.to(device)).squeeze()
            val_loss += criterion(outputs, labels.to(device)).item()
    if (epoch + 1) % 5 == 0:
        print(
            f"Oracle Epoch [{epoch + 1}/{ORACLE_EPOCHS}], Train Loss: {train_loss_epoch / len(oracle_on_train_loader):.4f}, Val Loss: {val_loss / len(oracle_on_val_loader):.4f}")
print("--- Oracle for ON training finished. ---")

# --- 训练 Oracle for off ---
print("\n--- Training Oracle for OFF score ---")
X_train_e, X_val_e, y_train_e, y_val_e = train_test_split(one_hot_sequences, off_scores, test_size=0.2,
                                                          random_state=42)
oracle_off_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_e), torch.from_numpy(y_train_e)),
                                     batch_size=32, shuffle=True)
oracle_off_val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_e), torch.from_numpy(y_val_e)),
                                   batch_size=32)
real_oracle_off = OracleCNN().to(device)
oracle_off_optimizer = optim.Adam(real_oracle_off.parameters(), lr=1e-3)
for epoch in range(ORACLE_EPOCHS):
    real_oracle_off.train()
    train_loss_epoch = 0
    for seqs, labels in oracle_off_train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        oracle_off_optimizer.zero_grad()
        outputs = real_oracle_off(seqs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        oracle_off_optimizer.step()
        train_loss_epoch += loss.item()
    real_oracle_off.eval()
    val_loss = 0
    with torch.no_grad():
        for seqs, labels in oracle_off_val_loader:
            outputs = real_oracle_off(seqs.to(device)).squeeze()
            val_loss += criterion(outputs, labels.to(device)).item()
    if (epoch + 1) % 5 == 0:
        print(
            f"Oracle Epoch [{epoch + 1}/{ORACLE_EPOCHS}], Train Loss: {train_loss_epoch / len(oracle_off_train_loader):.4f}, Val Loss: {val_loss / len(oracle_off_val_loader):.4f}")
print("--- Oracle for OFF training finished. ---")
torch.cuda.empty_cache()

on_mean, on_std = torch.tensor(on_scores.mean(), device=device), torch.tensor(on_scores.std(), device=device)
off_mean, off_std = torch.tensor(off_scores.mean(), device=device), torch.tensor(off_scores.std(),
                                                                                 device=device)


def multi_objective_reward_oracle(sequences_n_c_l):
    pred_on = real_oracle_on(sequences_n_c_l).squeeze()
    pred_off = real_oracle_off(sequences_n_c_l).squeeze()

    norm_on = (pred_on - on_mean) / on_std
    norm_off = (pred_off - off_mean) / off_std

    return W_on * norm_on - W_off * norm_off


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


def guided_sampling(diffusion_model, oracle_model, batch_size, guidance_scale=0.01, track_gradients=False):
    x_t = torch.randn((batch_size, NUCLEOTIDES, SEQUENCE_LENGTH), device=device)
    gradient_logs = []

    iterator = tqdm(reversed(range(diffusion_model.T)), desc=f"Sampling (Scale={guidance_scale})",
                    total=diffusion_model.T, leave=False)

    for t in iterator:
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # 1. 计算引导梯度
        grad = torch.zeros_like(x_t)
        if guidance_scale > 1e-6:
            with torch.enable_grad():
                x_t.requires_grad_()
                scores = oracle_model(x_t).sum()
                g = torch.autograd.grad(scores, x_t)[0]

                if track_gradients:
                    grad_norm = g.norm().item() / batch_size
                    gradient_logs.append(grad_norm)

                grad = g.clamp(-1, 1)
        elif track_gradients:
            gradient_logs.append(0)

        # 2. 扩散去噪步骤
        with torch.no_grad():
            predicted_noise = diffusion_model.model(x_t.detach(), t_tensor.float())
            alpha_t = diffusion_model.alphas[t]
            alpha_t_cumprod = diffusion_model.alphas_cumprod[t]

            guided_noise = predicted_noise - torch.sqrt(1. - alpha_t_cumprod) * grad * guidance_scale

            x_t_minus_1_mean = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod) * guided_noise) / torch.sqrt(
                alpha_t)

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t_minus_1_mean + torch.sqrt(diffusion_model.betas[t]) * noise
            else:
                x_t = x_t_minus_1_mean

    final_indices = torch.argmax(x_t, dim=1)
    final_seqs = F.one_hot(final_indices, num_classes=NUCLEOTIDES).float().permute(0, 2, 1)

    if track_gradients:
        return final_seqs, gradient_logs
    return final_seqs


def generate_sequences_in_batches(diffusion_model, oracle_model, total_samples, batch_size, guidance_scale=0.0):
    generated_seqs_list = []
    num_batches = math.ceil(total_samples / batch_size)
    print(f"Generating {total_samples} samples (Scale: {guidance_scale})...")

    for _ in range(num_batches):
        current_batch_size = min(batch_size, total_samples - sum(len(b) for b in generated_seqs_list))
        if current_batch_size <= 0: break

        batch_seqs = guided_sampling(diffusion_model, oracle_model, current_batch_size, guidance_scale,
                                     track_gradients=False)
        generated_seqs_list.append(batch_seqs.cpu())
        torch.cuda.empty_cache()

    return torch.cat(generated_seqs_list, dim=0).to(device)


# --- 生成数据 ---
N_EVAL_SAMPLES = 200
BATCH_SIZE = 32
GUIDANCE_STRENGTH = 0.5

# 1. Baseline Generation
baseline_diffusion_seqs = generate_sequences_in_batches(diffusion, multi_objective_reward_oracle, N_EVAL_SAMPLES,
                                                        BATCH_SIZE, 0.0)

# 2. RL-Guided Generation
rl_diffusion_seqs = generate_sequences_in_batches(diffusion, multi_objective_reward_oracle, N_EVAL_SAMPLES, BATCH_SIZE,
                                                  GUIDANCE_STRENGTH)


# 3. Random Generation (for Comparison)
def generate_random_sequences(num, length):
    return torch.nn.functional.one_hot(torch.randint(0, 4, (num, length)), num_classes=4).float().permute(0, 2, 1).to(
        device)


random_seqs_one_hot = generate_random_sequences(N_EVAL_SAMPLES, SEQUENCE_LENGTH)

# ==============================================================================
# 第6部分: 综合评估与可视化
# ==============================================================================
print("\n--- Part 6: Comprehensive Evaluation & Visualization ---")

# 辅助函数：转换 One-Hot 为字符串
def one_hot_to_strings(one_hot_tensor):
    if isinstance(one_hot_tensor, torch.Tensor):
        one_hot_tensor = one_hot_tensor.detach().cpu()
    indices = torch.argmax(one_hot_tensor.permute(0, 2, 1), dim=2)
    return ["".join([idx_to_char.get(i.item(), 'N') for i in seq]) for seq in indices]

# 转换所有序列
real_seqs_eval = real_seq_strings[:N_EVAL_SAMPLES]
baseline_seqs_str = one_hot_to_strings(baseline_diffusion_seqs)
rl_seqs_str = one_hot_to_strings(rl_diffusion_seqs)
random_seqs_str = one_hot_to_strings(random_seqs_one_hot)

# 计算分数
print("Calculating scores for all groups...")
with torch.no_grad():
    def get_scores(seqs_oh):
        on = real_oracle_on(seqs_oh).cpu().numpy().flatten()
        off = real_oracle_off(seqs_oh).cpu().numpy().flatten()
        comp = multi_objective_reward_oracle(seqs_oh).cpu().numpy().flatten()
        return on, off, comp

    on_real, off_real, comp_real = get_scores(torch.from_numpy(one_hot_sequences[:N_EVAL_SAMPLES]).to(device))
    on_base, off_base, comp_base = get_scores(baseline_diffusion_seqs)
    on_rl, off_rl, comp_rl = get_scores(rl_diffusion_seqs)
    on_rand, off_rand, comp_rand = get_scores(random_seqs_one_hot)

# ------------------------------------------------------------------------------
# 6.1 多目标分布散点图
# ------------------------------------------------------------------------------
print("Plotting Multi-Objective Landscape...")
plt.figure(figsize=(10, 8))
plt.scatter(off_rand, on_rand, c='gray', alpha=0.3, label='Random', s=20)
plt.scatter(off_real, on_real, c='blue', alpha=0.4, label='Real Data', s=20)
plt.scatter(off_base, on_base, c='orange', alpha=0.5, label='Baseline', s=30)
plt.scatter(off_rl, on_rl, c='red', alpha=0.6, label='RL-Guided', s=30)

best_idx = np.argmax(comp_rl)
plt.scatter(off_rl[best_idx], on_rl[best_idx], c='black', marker='*', s=200, label='Best Generated')

plt.xlabel("Predicted OFF (Lower is Better)")
plt.ylabel("Predicted ON Score (Higher is Better)")
plt.title(f"Multi-Objective Optimization Landscape\n(Maximize ON, Minimize OFF)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(output_dir, "multi_objective_scatter.png"), dpi=300)
plt.show()

# ------------------------------------------------------------------------------
# 6.2 小提琴图 (包含 ON, OFF, Composite) [修改点：补全所有小提琴图]
# ------------------------------------------------------------------------------
print("Plotting Violin Plots for ON, OFF, and Composite Scores...")

# (1) ON Score
df_on = pd.DataFrame({
    'Group': ['Real'] * len(on_real) + ['Random'] * len(on_rand) + ['Baseline'] * len(on_base) + ['RL-Guided'] * len(on_rl),
    'ON Score': np.concatenate([on_real, on_rand, on_base, on_rl])
})
plt.figure(figsize=(10, 6))
sns.violinplot(x='Group', y='ON Score', data=df_on, palette="viridis")
plt.title("Distribution of Predicted ON Scores (Higher is Better)")
plt.savefig(os.path.join(output_dir, "on_score_dist.png"), dpi=300)
plt.show()

# (2) OFF Score
df_off = pd.DataFrame({
    'Group': ['Real'] * len(off_real) + ['Random'] * len(off_rand) + ['Baseline'] * len(off_base) + ['RL-Guided'] * len(off_rl),
    'OFF Score': np.concatenate([off_real, off_rand, off_base, off_rl])
})
plt.figure(figsize=(10, 6))
sns.violinplot(x='Group', y='OFF Score', data=df_off, palette="magma")
plt.title("Distribution of Predicted OFF Scores (Lower is Better)")
plt.savefig(os.path.join(output_dir, "off_score_dist.png"), dpi=300)
plt.show()

# (3) Composite Score
df_comp = pd.DataFrame({
    'Group': ['Real'] * len(comp_real) + ['Random'] * len(comp_rand) + ['Baseline'] * len(comp_base) + ['RL-Guided'] * len(comp_rl),
    'Composite Score': np.concatenate([comp_real, comp_rand, comp_base, comp_rl])
})
plt.figure(figsize=(10, 6))
sns.violinplot(x='Group', y='Composite Score', data=df_comp, palette="muted")
plt.title("Composite Reward Distribution (Weighted ON & OFF)")
plt.savefig(os.path.join(output_dir, "composite_reward_dist.png"), dpi=300)
plt.show()

# ------------------------------------------------------------------------------
# 6.3 梯度动态可视化
# ------------------------------------------------------------------------------
print("Visualizing Gradient Dynamics...")
_, grad_logs = guided_sampling(diffusion, multi_objective_reward_oracle, batch_size=16,
                               guidance_scale=GUIDANCE_STRENGTH, track_gradients=True)

plt.figure(figsize=(10, 5))
plt.plot(grad_logs, label='Gradient Norm', color='purple')
plt.xlabel('Denoising Steps (T -> 0)')
plt.ylabel('Average Gradient Norm')
plt.title(f'RL Guidance Gradient Dynamics (Scale={GUIDANCE_STRENGTH})')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir, "gradient_dynamics.png"), dpi=300)
plt.show()


# ------------------------------------------------------------------------------
# 6.4 Top-5 序列分析与 Logo
# ------------------------------------------------------------------------------
def analyze_top_k(seqs_oh, seqs_str, scores, name, k=5):
    # 排序
    packed = sorted(list(zip(seqs_str, scores, range(len(scores)))), key=lambda x: x[1], reverse=True)[:k]
    print(f"\nTop {k} Sequences for {name}:")
    for i, (s, sc, _) in enumerate(packed):
        print(f"  {i + 1}. Score: {sc:.4f} | Seq: {s}")

    # 提取 Tensor 绘制 Logo
    top_oh = torch.stack([seqs_oh[x[2]] for x in packed])
    plot_logo(top_oh, title=f"Top {k} Sequences - {name}")


print("\n--- Generating Sequence Logos for Top Samples ---")
analyze_top_k(baseline_diffusion_seqs, baseline_seqs_str, comp_base, "Baseline")
analyze_top_k(rl_diffusion_seqs, rl_seqs_str, comp_rl, "RL-Guided")

# ------------------------------------------------------------------------------
# 6.5 GC Content
# ------------------------------------------------------------------------------
print("\nPlotting GC Content...")
def get_gc(seqs): return [(s.count('G') + s.count('C')) / len(s) for s in seqs]

sns.kdeplot(get_gc(real_seqs_eval), label='Real', fill=True)
sns.kdeplot(get_gc(rl_seqs_str), label='RL-Guided', fill=True)
plt.title("GC Content Distribution")
plt.legend()
plt.savefig(os.path.join(output_dir, "gc_dist.png"), dpi=300)
plt.show()

# ------------------------------------------------------------------------------
# 6.6 增强版 t-SNE
# ------------------------------------------------------------------------------
print("\nRunning t-SNE Visualization...")
def get_embs(data_oh):
    embs = []
    with torch.no_grad():
        # 使用 real_oracle_on 的特征层作为 Embedding
        for i in range(0, data_oh.shape[0], 32):
            batch = data_oh[i:i + 32].to(device)
            # 使用 on 预测器的中间层
            embs.append(real_oracle_on.get_embedding(batch).cpu().numpy())
    return np.concatenate(embs, axis=0)

# 获取 Embedding
emb_real = get_embs(torch.from_numpy(one_hot_sequences[:N_EVAL_SAMPLES]))
emb_rand = get_embs(random_seqs_one_hot)
emb_base = get_embs(baseline_diffusion_seqs)
emb_rl = get_embs(rl_diffusion_seqs)

all_embs = np.concatenate([emb_rand, emb_real, emb_base, emb_rl], axis=0)
labels = ['Random'] * len(emb_rand) + ['Real'] * len(emb_real) + ['Baseline'] * len(emb_base) + ['RL-Guided'] * len(
    emb_rl)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
tsne_res = tsne.fit_transform(all_embs)

df_tsne = pd.DataFrame({'Dim 1': tsne_res[:, 0], 'Dim 2': tsne_res[:, 1], 'Type': labels})
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_tsne, x='Dim 1', y='Dim 2', hue='Type', style='Type', alpha=0.7, s=60)
plt.title("t-SNE of Sequence Embeddings (Latent Space)")
plt.savefig(os.path.join(output_dir, "tsne_enhanced.png"), dpi=300)
plt.show()

# ==============================================================================
# 第7部分: 结果保存 (Saving)
# ==============================================================================
print("\n--- Part 7: Saving Results to File ---")

def save_to_file(filename, seqs, on, off, comp):
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        f.write("Sequence,ON_Score,OFF_Score,Composite_Score\n")
        for i in range(len(seqs)):
            f.write(f"{seqs[i]},{on[i]:.4f},{off[i]:.4f},{comp[i]:.4f}\n")
    print(f"Saved {filename}")

save_to_file("sequences_baseline.txt", baseline_seqs_str, on_base, off_base, comp_base)
save_to_file("sequences_rl_guided.txt", rl_seqs_str, on_rl, off_rl, comp_rl)

print("\nAll evaluation tasks completed successfully!")