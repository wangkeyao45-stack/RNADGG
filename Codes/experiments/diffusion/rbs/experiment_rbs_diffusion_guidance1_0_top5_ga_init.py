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
# 第0部分: 环境设置
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sns.set_theme(style="whitegrid")
output_dir = "output_plots_diffusion_rbs_1.0_top5_gai"
os.makedirs(output_dir, exist_ok=True)
print(f"所有图表将被保存在 '{output_dir}/' 文件夹中。")


def plot_logo(one_hot_seqs, title=''):
    """
    使用logomaker库绘制专业的序列标识图
    """
    if isinstance(one_hot_seqs, torch.Tensor):
        one_hot_seqs = one_hot_seqs.detach().cpu().numpy()

    if one_hot_seqs.shape[-1] != 4:
        one_hot_seqs = np.transpose(one_hot_seqs, (0, 2, 1))

    pwm = np.mean(one_hot_seqs, axis=0)
    pwm_df = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])

    seq_length = pwm_df.shape[0]
    fig_width = max(10, seq_length * 0.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 2.5))

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    logo = logomaker.Logo(pwm_df,
                          ax=ax,
                          shade_below=0.5,
                          fade_below=0.5,
                          font_name='sans-serif',
                          color_scheme='colorblind_safe')

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

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"logo_{sanitized_title}.png"), dpi=300, bbox_inches='tight')
    plt.show()


# ==============================================================================
# 第1部分: PyTorch 模型定义
# ==============================================================================
SEQUENCE_LENGTH = 17
NUCLEOTIDES = 4
CSV_FILE = "rbs_data.csv"


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

    def get_embedding(self, seq): return self.fc_net[0](self.conv_net(seq))


# ==============================================================================
# 第2, 3, 4部分: 数据与训练
# ==============================================================================
print("--- Part 2: Loading Data ---")
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
NUM_SAMPLES_TO_USE = 280000
df_subset = df.head(NUM_SAMPLES_TO_USE)
sequences = df_subset['序列'].tolist()
rl_scores = df_subset['rl'].values.astype(np.float32)
processed_data = [(one_hot_encode(s), r, s) for s, r in zip(sequences, rl_scores) if one_hot_encode(s) is not None]
one_hot_sequences = np.array([item[0] for item in processed_data])
scores = np.array([item[1] for item in processed_data])
real_seq_strings = [item[2] for item in processed_data]
print(f"Data loaded. Shape: {one_hot_sequences.shape}")

print("\n--- Part 3: Training Real Oracle ---")
X_train, X_val, y_train, y_val = train_test_split(one_hot_sequences, scores, test_size=0.2, random_state=42)
oracle_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32,
                                 shuffle=True)
oracle_val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=32)
real_oracle = OracleCNN().to(device)
oracle_optimizer, criterion = optim.Adam(real_oracle.parameters(), lr=1e-3), nn.MSELoss()
ORACLE_EPOCHS = 50
for epoch in range(ORACLE_EPOCHS):
    real_oracle.train()
    for seqs, labels in oracle_train_loader:
        oracle_optimizer.zero_grad()
        loss = criterion(real_oracle(seqs.to(device)).squeeze(), labels.to(device))
        loss.backward()
        oracle_optimizer.step()
    if (epoch + 1) % 10 == 0: print(f"Oracle Epoch {epoch + 1}/{ORACLE_EPOCHS} done.")
real_oracle.eval()

print("\n--- Part 4: Training Diffusion ---")
diffusion_train_loader = DataLoader(TensorDataset(torch.from_numpy(one_hot_sequences)), batch_size=64, shuffle=True)
unet = UNet().to(device)
diffusion = Diffusion(unet)
diffusion_optimizer = optim.Adam(unet.parameters(), lr=1e-4)
DIFFUSION_EPOCHS = 100
for epoch in range(DIFFUSION_EPOCHS):
    pbar = tqdm(diffusion_train_loader, desc=f"Diff Epoch {epoch + 1}/{DIFFUSION_EPOCHS}", leave=False)
    for i, (real_seqs,) in enumerate(pbar):
        diffusion_optimizer.zero_grad()
        loss = diffusion.train_step(real_seqs.to(device))
        loss.backward()
        diffusion_optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
print("Diffusion training finished.")
torch.cuda.empty_cache()

# ==============================================================================
# 第5部分: 定义生成函数
# ==============================================================================
# ==============================================================================
## ==============================================================================
# 第5部分: 定义生成函数 (修复版：连接 Robust Sampling)
# ==============================================================================
print("\n--- Part 5: Generation Functions (Fixed) ---")

def robust_guided_sampling(diffusion_model, oracle_model, batch_size, guidance_scale=10.0, start_guidance_step=200):
    """
    改进版采样函数：支持 Softmax Trick 和 梯度归一化
    """
    x_t = torch.randn((batch_size, NUCLEOTIDES, SEQUENCE_LENGTH), device=device)

    for t in tqdm(reversed(range(diffusion_model.T)), desc="Robust Sampling", total=diffusion_model.T, leave=False):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        grad = torch.zeros_like(x_t)

        # 只在去噪的后半段开启引导
        if t < start_guidance_step and guidance_scale != 0:
            with torch.enable_grad():
                x_in = x_t.detach().requires_grad_()
                x_soft = F.softmax(x_in, dim=1) # Softmax Trick
                scores = oracle_model(x_soft).sum()
                g = torch.autograd.grad(scores, x_in)[0]

                # 梯度归一化
                if torch.norm(g) > 1e-8:
                    g = g / (torch.norm(g) + 1e-8)
                grad = g

        with torch.no_grad():
            predicted_noise = diffusion_model.model(x_t, t_tensor.float())
            alpha_t = diffusion_model.alphas[t]
            alpha_t_cumprod = diffusion_model.alphas_cumprod[t]
            beta_t = diffusion_model.betas[t]

            # 最大化分数 = 减去梯度方向
            guided_noise = predicted_noise - torch.sqrt(1. - alpha_t_cumprod) * grad * guidance_scale

            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod)
            x_t_minus_1_mean = coef1 * (x_t - coef2 * guided_noise)

            if t > 0:
                sigma = torch.sqrt(beta_t)
                x_t = x_t_minus_1_mean + sigma * torch.randn_like(x_t)
            else:
                x_t = x_t_minus_1_mean

    return F.one_hot(torch.argmax(x_t, dim=1), num_classes=NUCLEOTIDES).float().permute(0, 2, 1)

def guided_sampling_target(diffusion_model, oracle_model, batch_size, target_value, guidance_scale=1.0):
    """
    定值生成函数
    """
    x_t = torch.randn((batch_size, NUCLEOTIDES, SEQUENCE_LENGTH), device=device)
    target_tensor = torch.full((batch_size, 1), target_value, device=device).float()

    for t in tqdm(reversed(range(diffusion_model.T)), desc=f"Target={target_value}", total=diffusion_model.T, leave=False):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # 简单定值引导（不做复杂归一化，使用 MSE Loss）
        with torch.enable_grad():
            x_t.requires_grad_()
            preds = oracle_model(x_t)
            loss = -((preds - target_tensor) ** 2).sum() # 梯度上升: 减小 loss
            grad = torch.autograd.grad(loss, x_t)[0]
            grad = grad.clamp(-1, 1)

        with torch.no_grad():
            predicted_noise = diffusion_model.model(x_t.detach(), t_tensor.float())
            alpha_t, alpha_t_cumprod = diffusion_model.alphas[t], diffusion_model.alphas_cumprod[t]
            guided_noise = predicted_noise - torch.sqrt(1. - alpha_t_cumprod) * grad * guidance_scale

            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod)
            x_t_minus_1_mean = coef1 * (x_t - coef2 * guided_noise)

            if t > 0:
                x_t = x_t_minus_1_mean + torch.sqrt(diffusion_model.betas[t]) * torch.randn_like(x_t)
            else:
                x_t = x_t_minus_1_mean

    return F.one_hot(torch.argmax(x_t, dim=1), num_classes=NUCLEOTIDES).float().permute(0, 2, 1)

def generate_sequences_in_batches(diffusion_model, oracle_model, total_samples, batch_size, guidance_scale=0.0, target_val=None):
    """
    [修复] 包装器现在正确调用 robust_guided_sampling
    """
    generated_seqs_list = []
    num_batches = math.ceil(total_samples / batch_size)

    for i in range(num_batches):
        current_batch_size = min(batch_size, total_samples - len(generated_seqs_list) * batch_size)
        if current_batch_size <= 0: break

        # 确保剩余数量正确计算
        needed = total_samples - sum([len(x) for x in generated_seqs_list])
        current_batch_size = min(batch_size, needed)

        if target_val is not None:
            batch_seqs = guided_sampling_target(diffusion_model, oracle_model, current_batch_size, target_val, guidance_scale)
        else:
            # [关键修复] 这里调用 robust_guided_sampling，并且设置 start_step
            # 注意：这里的 scale 和 start_step 可以写死或者作为参数传入，这里使用默认较优值
            start_step = 200 if guidance_scale > 0 else 0
            batch_seqs = robust_guided_sampling(diffusion_model, oracle_model, current_batch_size,
                                              guidance_scale=guidance_scale,
                                              start_guidance_step=start_step)

        generated_seqs_list.append(batch_seqs.cpu())
        torch.cuda.empty_cache()

    return torch.cat(generated_seqs_list, dim=0).to(device)

# 补充丢失的辅助函数 (Part 6/7 中丢失的)
def one_hot_to_strings(one_hot_tensor):
    if isinstance(one_hot_tensor, torch.Tensor): one_hot_tensor = one_hot_tensor.detach().cpu().numpy()
    if one_hot_tensor.shape[1] == 4: one_hot_tensor = np.transpose(one_hot_tensor, (0, 2, 1))
    indices = np.argmax(one_hot_tensor, axis=2)
    return ["".join([idx_to_char.get(i, 'N') for i in seq]) for seq in indices]

def generate_random_sequences(num_seqs, seq_len):
    sequences_str = ["".join(np.random.choice(['A', 'C', 'G', 'T'], size=seq_len)) for _ in range(num_seqs)]
    one_hot_seqs = [one_hot_encode(seq) for seq in sequences_str if one_hot_encode(seq) is not None]
    if one_hot_seqs:
        return sequences_str, torch.from_numpy(np.array(one_hot_seqs)).float().to(device)
    else:
        return sequences_str, torch.tensor([]).to(device)

def save_sequences_to_txt(filename, sequences, scores=None):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        f.write("Sequence,Predicted_Score\n")
        for i, seq in enumerate(sequences):
            score_str = f"{scores[i]:.4f}" if scores is not None else "N/A"
            f.write(f"{seq},{score_str}\n")
    print(f"Saved {len(sequences)} sequences to {filepath}")


# ==============================================================================
# 第6 & 7部分: 综合评估 (已修复)
# ==============================================================================
print("\n--- Improved Part 7: Robust Evaluation ---")

EVAL_BATCH_SIZE = 64
N_SAMPLES = 256
GUIDANCE_VAL = 15.0 # 定义全局变量供后续使用

print("1. Generating Baseline (No Guidance)...")
seqs_base = robust_guided_sampling(diffusion, real_oracle, N_SAMPLES, guidance_scale=0.0)

print("2. Generating Robust Guided (Optimized)...")
seqs_optim = robust_guided_sampling(diffusion, real_oracle, N_SAMPLES, guidance_scale=GUIDANCE_VAL, start_guidance_step=200)

with torch.no_grad():
    scores_base = real_oracle(seqs_base).cpu().numpy().flatten()
    scores_optim = real_oracle(seqs_optim).cpu().numpy().flatten()

str_base = one_hot_to_strings(seqs_base)
str_optim = one_hot_to_strings(seqs_optim)

def calculate_diversity(seq_list):
    if len(seq_list) > 100: sample = np.random.choice(seq_list, 100, replace=False)
    else: sample = seq_list
    dists = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            dists.append(Levenshtein.distance(sample[i], sample[j]))
    return np.mean(dists) if dists else 0

div_base = calculate_diversity(str_base)
div_optim = calculate_diversity(str_optim)

print(f"\nResults Comparison:")
print(f"{'Metric':<15} | {'Baseline':<15} | {'Robust Guided':<15}")
print("-" * 50)
print(f"{'Mean Score':<15} | {np.mean(scores_base):.4f}          | {np.mean(scores_optim):.4f}")
print(f"{'Max Score':<15}  | {np.max(scores_base):.4f}          | {np.max(scores_optim):.4f}")
print(f"{'Diversity':<15} | {div_base:.2f}           | {div_optim:.2f}")

# 简单可视化
plt.figure(figsize=(10, 5))
sns.kdeplot(scores_base, label='Baseline', fill=True, color='gray')
sns.kdeplot(scores_optim, label='Robust Guided', fill=True, color='red')
plt.title("Score Distribution Shift")
plt.legend()
plt.savefig(os.path.join(output_dir, "robust_optimization_result.png"), dpi=300)
plt.show()

# ==============================================================================
# 第8部分: Top 5 分析 (修复版：使用正确的函数)
# ==============================================================================
print("\n--- Part 8: Top 5 Analysis ---")

def process_top_n(seqs_oh, seqs_str, name):
    if seqs_oh is None: return
    with torch.no_grad():
        preds = real_oracle(seqs_oh).cpu().numpy().flatten()
    packed = sorted(list(zip(seqs_str, preds, range(len(preds)))), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 {name}:")
    for i, (s, sc, _) in enumerate(packed): print(f"  {i + 1}. {sc:.4f}: {s}")
    top_oh = torch.stack([seqs_oh[x[2]] for x in packed])
    plot_logo(top_oh, title=f"Top 5 {name}")

N_SEARCH = 500
print(f"Generating {N_SEARCH} samples for Top-5 search...")

# Random
r_str, r_oh = generate_random_sequences(N_SEARCH, SEQUENCE_LENGTH)
process_top_n(r_oh, r_str, "Random")

# Baseline
b_oh = generate_sequences_in_batches(diffusion, real_oracle, N_SEARCH, 64, 0.0)
b_str = one_hot_to_strings(b_oh)
process_top_n(b_oh, b_str, "Baseline")

# RL-Guided (引用 GUIDANCE_VAL)
rl_oh = generate_sequences_in_batches(diffusion, real_oracle, N_SEARCH, 64, GUIDANCE_VAL)
rl_str = one_hot_to_strings(rl_oh)
process_top_n(rl_oh, rl_str, "RL-Guided")

# ==============================================================================
# 第9部分: 保存结果 (去掉了 crash 的梯度可视化)
# ==============================================================================
print("\n--- Part 9: Saving Results ---")
# 梯度可视化已移除，因为 Robust Sampling 不返回梯度日志

with torch.no_grad():
    scores_rand = real_oracle(r_oh).cpu().numpy().flatten() if len(r_oh) > 0 else []
    scores_base = real_oracle(b_oh).cpu().numpy().flatten()
    scores_rl = real_oracle(rl_oh).cpu().numpy().flatten()

save_sequences_to_txt("sequences_random.txt", r_str, scores_rand)
save_sequences_to_txt("sequences_baseline.txt", b_str, scores_base)
save_sequences_to_txt("sequences_rl_guided.txt", rl_str, scores_rl)

# ==============================================================================
# 第10部分: 定值生成验证 (修复：使用正确的 target_val 处理)
# ==============================================================================
print("\n--- Part 10: Target Value Generation Verification ---")

target_values = [0.4, 0.6, 0.8]

plt.figure(figsize=(10, 6))

for target in target_values:
    print(f"Generating sequences targeting RL Score = {target}...")

    # 增加 guidance_scale 到 2.0 或 5.0
    target_seqs_oh = generate_sequences_in_batches(
        diffusion, real_oracle,
        total_samples=100,
        batch_size=32,
        guidance_scale=2.0,
        target_val=target
    )

    with torch.no_grad():
        actual_scores = real_oracle(target_seqs_oh).cpu().numpy().flatten()

    mean_score = np.mean(actual_scores)
    print(f"  -> Target: {target}, Actual Mean: {mean_score:.4f}")
    sns.kdeplot(actual_scores, label=f'Target {target}', fill=True, alpha=0.3)

plt.title("Conditional Generation: Target vs Actual Score")
plt.legend()
plt.savefig(os.path.join(output_dir, "target_generation_verification.png"), dpi=300)
plt.show()

print("\nAll tasks completed successfully!")