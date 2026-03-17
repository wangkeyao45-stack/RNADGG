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
output_dir = "output_plots_diffusion_rbs_1.0_top5_hyp22"
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
CSV_FILE = "rbs_data_f.csv"


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
    def __init__(self, n_channels=128):  # 修改点: CH=128
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
    def __init__(self, noise_predictor_model, T=1000):  # 修改点: T=1000
        self.model = noise_predictor_model
        self.T = T
        self.betas = torch.linspace(1e-4, 0.02, T, device=device)  # 保持 linear 调度
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
            nn.Conv1d(NUCLEOTIDES, 128, kernel_size=7, padding='same'), nn.ReLU(), # 修改点: CH=128
            nn.Dropout(0.3), # 修改点: DR=0.3
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding='same'), nn.ReLU(),
            nn.Dropout(0.3), # 修改点: DR=0.3
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        self.fc_net = nn.Sequential(nn.Linear(256 * (SEQUENCE_LENGTH // 4), 64), nn.ReLU(), nn.Linear(64, 1))

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
NUM_SAMPLES_TO_USE = 310000
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
oracle_optimizer, criterion = optim.Adam(real_oracle.parameters(), lr=0.0005), nn.MSELoss() # 修改点: LR=0.0005
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
unet = UNet(n_channels=128).to(device) # 修改点: CH=128
diffusion = Diffusion(unet, T=1000) # 修改点: T=1000
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
print("\n--- Part 5: Generation Functions (Enhanced) ---")


def guided_sampling(diffusion_model, oracle_model, batch_size, guidance_scale=0.01, track_gradients=False):
    """
    RL引导采样 (最大化分数模式)
    """
    x_t = torch.randn((batch_size, NUCLEOTIDES, SEQUENCE_LENGTH), device=device)
    gradient_logs = []

    for t in tqdm(reversed(range(diffusion_model.T)), desc="Sampling (Max)", total=diffusion_model.T, leave=False):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        if guidance_scale != 0:
            with torch.enable_grad():
                x_t.requires_grad_()
                scores = oracle_model(x_t).sum()
                grad = torch.autograd.grad(scores, x_t)[0]

                if track_gradients:
                    grad_norm = grad.norm().item() / batch_size
                    gradient_logs.append(grad_norm)

                grad = grad.clamp(-1, 1)
        else:
            grad = torch.zeros_like(x_t)
            if track_gradients: gradient_logs.append(0)

        with torch.no_grad():
            predicted_noise = diffusion_model.model(x_t.detach(), t_tensor.float())
            alpha_t, alpha_t_cumprod = diffusion_model.alphas[t], diffusion_model.alphas_cumprod[t]

            guided_noise = predicted_noise - torch.sqrt(1. - alpha_t_cumprod) * grad * guidance_scale
            x_t_minus_1_mean = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod) * guided_noise) / torch.sqrt(
                alpha_t)

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t_minus_1_mean + torch.sqrt(diffusion_model.betas[t]) * noise
            else:
                x_t = x_t_minus_1_mean

    final_seqs = F.one_hot(torch.argmax(x_t, dim=1), num_classes=NUCLEOTIDES).float().permute(0, 2, 1)

    if track_gradients:
        return final_seqs, gradient_logs
    return final_seqs


def guided_sampling_target(diffusion_model, oracle_model, batch_size, target_value, guidance_scale=0.1):
    """
    定值生成：生成接近 target_value 分数的序列
    """
    x_t = torch.randn((batch_size, NUCLEOTIDES, SEQUENCE_LENGTH), device=device)
    target_tensor = torch.full((batch_size, 1), target_value, device=device).float()

    for t in tqdm(reversed(range(diffusion_model.T)), desc=f"Sampling Target={target_value}", total=diffusion_model.T,
                  leave=False):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        with torch.enable_grad():
            x_t.requires_grad_()
            preds = oracle_model(x_t)
            loss = -((preds - target_tensor) ** 2).sum()
            grad = torch.autograd.grad(loss, x_t)[0]
            grad = grad.clamp(-1, 1)

        with torch.no_grad():
            predicted_noise = diffusion_model.model(x_t.detach(), t_tensor.float())
            alpha_t, alpha_t_cumprod = diffusion_model.alphas[t], diffusion_model.alphas_cumprod[t]

            guided_noise = predicted_noise - torch.sqrt(1. - alpha_t_cumprod) * grad * guidance_scale
            x_t_minus_1_mean = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod) * guided_noise) / torch.sqrt(
                alpha_t)

            if t > 0:
                x_t = x_t_minus_1_mean + torch.sqrt(diffusion_model.betas[t]) * torch.randn_like(x_t)
            else:
                x_t = x_t_minus_1_mean

    return F.one_hot(torch.argmax(x_t, dim=1), num_classes=NUCLEOTIDES).float().permute(0, 2, 1)


def generate_sequences_in_batches(diffusion_model, oracle_model, total_samples, batch_size, guidance_scale=0.0,
                                  target_val=None):
    generated_seqs_list = []
    num_batches = math.ceil(total_samples / batch_size)
    mode_str = f"Target={target_val}" if target_val is not None else ("MaxRL" if guidance_scale > 0 else "Baseline")
    print(f"Generating {total_samples} samples [{mode_str}] (Guidance: {guidance_scale})...")

    for i in range(num_batches):
        current_batch_size = min(batch_size, total_samples - len(generated_seqs_list) * batch_size)
        remaining = total_samples - sum(len(b) for b in generated_seqs_list)
        current_batch_size = min(batch_size, remaining)
        if current_batch_size <= 0: break

        if target_val is not None:
            batch_seqs = guided_sampling_target(diffusion_model, oracle_model, current_batch_size, target_val,
                                                guidance_scale)
        else:
            batch_seqs = guided_sampling(diffusion_model, oracle_model, current_batch_size, guidance_scale)

        generated_seqs_list.append(batch_seqs.cpu())
        torch.cuda.empty_cache()

    return torch.cat(generated_seqs_list, dim=0).to(device)


def save_sequences_to_txt(filename, sequences, scores=None):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        f.write("Sequence,Predicted_Score\n")
        for i, seq in enumerate(sequences):
            score_str = f"{scores[i]:.4f}" if scores is not None else "N/A"
            f.write(f"{seq},{score_str}\n")
    print(f"Saved {len(sequences)} sequences to {filepath}")


def calculate_ic_profile(one_hot_tensor):
    """
    计算序列每个位置的信息含量 (Information Content)
    公式: R_i = 2 - (-sum(p * log2(p)))
    Input: (N, 4, L) tensor or array
    Output: (L,) array containing IC bits per position
    """
    if isinstance(one_hot_tensor, np.ndarray):
        one_hot_tensor = torch.from_numpy(one_hot_tensor)

    # 确保在 CPU 上计算以方便绘图
    one_hot_tensor = one_hot_tensor.float().cpu()

    # 1. 计算位置概率矩阵 (Position Probability Matrix, PPM/PWM)
    # Shape: (4, L)
    ppm = torch.mean(one_hot_tensor, dim=0)

    # 2. 计算香农熵 (Shannon Entropy)
    # 添加极小值 1e-12 避免 log2(0) 导致 NaN
    entropy = -torch.sum(ppm * torch.log2(ppm + 1e-12), dim=0)

    # 3. 计算信息含量 (IC)
    # DNA/RNA 有 4 种碱基，最大熵为 log2(4) = 2 bits
    max_entropy = 2.0
    ic_profile = max_entropy - entropy

    return ic_profile.numpy()
# ==============================================================================
# 第6 & 7部分: 综合评估
# ==============================================================================
print("\n--- Part 7: Comprehensive Evaluation ---")


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




# 参数设置
N_EVAL_SAMPLES = 200
BATCH_SIZE = 32
GUIDANCE = 1.0

# --- 修改点 1: 随机选择 200 条真实序列 ---
indices = np.random.permutation(len(processed_data))[:N_EVAL_SAMPLES]
real_seqs_eval = [real_seq_strings[i] for i in indices]
real_one_hot_eval = torch.from_numpy(one_hot_sequences[indices]).float().to(device)
print(f"Randomly selected {N_EVAL_SAMPLES} real sequences for evaluation.")

# 生成对比数据
print("Generating Random...")
random_seqs_str, random_seqs_one_hot = generate_random_sequences(N_EVAL_SAMPLES, SEQUENCE_LENGTH)

print("Generating Baseline...")
baseline_diffusion_seqs = generate_sequences_in_batches(diffusion, real_oracle, N_EVAL_SAMPLES, BATCH_SIZE, 0.0)
baseline_seqs_str = one_hot_to_strings(baseline_diffusion_seqs)

print("Generating RL-Guided...")
rl_diffusion_seqs = generate_sequences_in_batches(diffusion, real_oracle, N_EVAL_SAMPLES, BATCH_SIZE, GUIDANCE)
rl_seqs_str = one_hot_to_strings(rl_diffusion_seqs)

print("\n--- Calculating Metrics ---")

def calc_lev(gen, real):
    return sum([min([Levenshtein.distance(g, r) for r in real]) for g in gen]) / len(gen)

dist_rand = calc_lev(random_seqs_str, real_seqs_eval)
dist_base = calc_lev(baseline_seqs_str, real_seqs_eval)
dist_rl = calc_lev(rl_seqs_str, real_seqs_eval)
print(f"Levenshtein: Rand={dist_rand:.2f}, Base={dist_base:.2f}, RL={dist_rl:.2f}")

def get_kmer_dist(seqs, k=4):
    cnt = Counter([s[i:i + k] for s in seqs for i in range(len(s) - k + 1)])
    total = sum(cnt.values())
    return {k: v / total for k, v in cnt.items()}

real_dist = get_kmer_dist(real_seqs_eval)
all_k = sorted(list(set(real_dist.keys()) | set(get_kmer_dist(random_seqs_str).keys())))

def calc_jsd(seqs):
    d = get_kmer_dist(seqs)
    p = np.array([real_dist.get(k, 0) for k in all_k])
    q = np.array([d.get(k, 0) for k in all_k])
    return jensenshannon(p, q, base=2.0)

jsd_rand = calc_jsd(random_seqs_str)
jsd_base = calc_jsd(baseline_seqs_str)
jsd_rl = calc_jsd(rl_seqs_str)
print(f"JSD: Rand={jsd_rand:.4f}, Base={jsd_base:.4f}, RL={jsd_rl:.4f}")

def get_embs(data):
    embs = []
    with torch.no_grad():
        for i in range(0, data.shape[0], 32):
            batch_emb = real_oracle.get_embedding(data[i:i+32])
            embs.append(batch_emb.detach().cpu().numpy())
    return np.concatenate(embs, axis=0)

real_embs = get_embs(real_one_hot_eval)
base_embs = get_embs(baseline_diffusion_seqs)
rl_embs = get_embs(rl_diffusion_seqs)

def calc_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

mu_real, sig_real = np.mean(real_embs, axis=0), np.cov(real_embs, rowvar=False)
mu_base, sig_base = np.mean(base_embs, axis=0), np.cov(base_embs, rowvar=False)
mu_rl, sig_rl = np.mean(rl_embs, axis=0), np.cov(rl_embs, rowvar=False)

fid_base = calc_fid(mu_real, sig_real, mu_base, sig_base)
fid_rl = calc_fid(mu_real, sig_real, mu_rl, sig_rl)
print(f"FID: Base={fid_base:.4f}, RL={fid_rl:.4f}")
print("\n--- Calculating Information Content (IC) Profile ---")

# 计算各组的 IC 分布
ic_real = calculate_ic_profile(real_one_hot_eval)
ic_rand = calculate_ic_profile(random_seqs_one_hot)
ic_base = calculate_ic_profile(baseline_diffusion_seqs)
ic_rl = calculate_ic_profile(rl_diffusion_seqs)

# 计算平均 IC (作为整体保守性的单一指标)
mean_ic_real = np.mean(ic_real)
mean_ic_rl = np.mean(ic_rl)
print(f"Mean IC per position: Real={mean_ic_real:.4f}, RL-Guided={mean_ic_rl:.4f}")

# === 可视化 IC 对比图 ===
plt.figure(figsize=(12, 6))

# 设置 x 轴位置 (1 到 17)
x_axis = range(1, SEQUENCE_LENGTH + 1)

# 绘制曲线
plt.plot(x_axis, ic_real, 'k-', linewidth=2.5, label='Real Biological Data', alpha=0.8)
plt.plot(x_axis, ic_rand, ':', color='gray', linewidth=2, label='Random (Noise)', alpha=0.6)
plt.plot(x_axis, ic_base, '--', color='blue', linewidth=2, label='Baseline Diffusion', alpha=0.7)
plt.plot(x_axis, ic_rl, '-', color='red', linewidth=2, label='RL-Guided Diffusion', alpha=0.8)

# 标注 Shine-Dalgarno 区域 (假设在序列后半段，通常在 -13 到 -8 左右，这里仅作示意，根据实际数据调整)
# 如果您的序列是 RBS，通常在第 8-13 位附近会有高 IC 峰值
plt.axhline(y=2.0, color='k', linestyle='-', alpha=0.1) # Max bits
plt.axhline(y=0.0, color='k', linestyle='-', alpha=0.1) # Min bits

plt.title("Information Content (Conservation) Profile per Position", fontsize=14)
plt.xlabel("Position in Sequence", fontsize=12)
plt.ylabel("Information Content (Bits)", fontsize=12)
plt.ylim(-0.1, 2.1)
plt.xticks(x_axis)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

# 保存图片
plt.savefig(os.path.join(output_dir, "ic_profile_comparison.png"), dpi=300)

plt.show()

# 简单的峰值检测 (检测 Shine-Dalgarno 核心 AGGAGG 是否被捕获)
# 假设 SD 序列通常产生高 IC 值
peak_indices = np.where(ic_rl > 1.0)[0]
print(f"High Conservation Positions (IC > 1.0) in RL-Guided: {[i+1 for i in peak_indices]}")
print("\n--- Plotting Reward Distribution ---")
with torch.no_grad():
    p_real = real_oracle(real_one_hot_eval).cpu().numpy().flatten()
    p_rand = real_oracle(random_seqs_one_hot).cpu().numpy().flatten()
    p_base = real_oracle(baseline_diffusion_seqs).cpu().numpy().flatten()
    p_rl = real_oracle(rl_diffusion_seqs).cpu().numpy().flatten()

df_viol = pd.DataFrame({
    'Group': ['Real'] * len(p_real) + ['Random'] * len(p_rand) + ['Baseline'] * len(p_base) + ['RL-Guided'] * len(p_rl),
    'Score': np.concatenate([p_real, p_rand, p_base, p_rl])
})
plt.figure(figsize=(10, 6))
sns.violinplot(x='Group', y='Score', hue='Group', data=df_viol, palette="muted", legend=False)
plt.title("Predicted Reward Distribution")
plt.savefig(os.path.join(output_dir, "reward_dist.png"), dpi=300)
plt.show()
# ==============================================================================
# 第8部分: 序列分析与全量可视化 (优化版)
# ==============================================================================
print("\n--- Part 8: Sequence Analysis (Full & Top 5) ---")

# 我们提高搜索样本量到 500，以获得更好的 Top 5
N_SEARCH = 500


def analyze_group(name, guidance=0.0):
    print(f"\nAnalyzing {name} (Guidance: {guidance})...")

    # 1. 生成/准备数据
    if name == "Random":
        s_str, s_oh = generate_random_sequences(N_SEARCH, SEQUENCE_LENGTH)
    else:
        # 统一使用 N_SEARCH 进行生成
        s_oh = generate_sequences_in_batches(diffusion, real_oracle, N_SEARCH, BATCH_SIZE, guidance)
        s_str = one_hot_to_strings(s_oh)

    # 2. 全量序列可视化 (200~500条)
    plot_logo(s_oh, title=f"Full Set: {name} ({len(s_str)} seqs)")

    # 3. 筛选并可视化 Top 5
    with torch.no_grad():
        # 确保数据在正确的设备上进行预测
        preds = real_oracle(s_oh.to(device)).cpu().numpy().flatten()

    packed = sorted(list(zip(s_str, preds, range(len(preds)))), key=lambda x: x[1], reverse=True)[:5]

    print(f"Top 5 {name} Scores:")
    for i, (s, sc, _) in enumerate(packed):
        print(f"  {i + 1}. {sc:.4f}: {s}")

    # 从生成的大组中提取对应的 one-hot
    top_oh = torch.stack([s_oh.cpu()[x[2]] for x in packed])
    plot_logo(top_oh, title=f"Top 5 {name}")

    return s_oh, s_str


# --- 执行分析 ---

# 1. 随机真实序列 (使用 Part 7 中随机选出的 200 条即可，无需重选)
plot_logo(real_one_hot_eval, title=f"Full Set: Random Real ({len(real_seqs_eval)} seqs)")

# 2. 随机生成对照
analyze_group("Random")

# 3. 扩散模型 Baseline (无引导)
analyze_group("Baseline", guidance=0.0)

# 4. 扩散模型 RL-Guided (强引导)
analyze_group("RL-Guided", guidance=GUIDANCE)

print("\n--- Part 9: Advanced Analysis & Saving ---")
vis_batch_size = 16
_, grad_logs = guided_sampling(diffusion, real_oracle, batch_size=vis_batch_size, guidance_scale=GUIDANCE,
                               track_gradients=True)

plt.figure(figsize=(10, 5))
plt.plot(grad_logs, label='Gradient Norm', color='purple')
plt.xlabel('Denoising Steps (T -> 0)')
plt.ylabel('Average Gradient Norm')
plt.title(f'Guidance Gradient Dynamics (Scale={GUIDANCE})')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir, "gradient_dynamics.png"), dpi=300)
plt.show()

with torch.no_grad():
    scores_rand = real_oracle(random_seqs_one_hot).cpu().numpy().flatten()
    scores_base = real_oracle(baseline_diffusion_seqs).cpu().numpy().flatten()
    scores_rl = real_oracle(rl_diffusion_seqs).cpu().numpy().flatten()

save_sequences_to_txt("sequences_random.txt", random_seqs_str, scores_rand)
save_sequences_to_txt("sequences_baseline.txt", baseline_seqs_str, scores_base)
save_sequences_to_txt("sequences_rl_guided.txt", rl_seqs_str, scores_rl)

print("\n--- Part 10: Target Value Generation Verification ---")
target_values = [0.2, 0.4, 0.6]
plt.figure(figsize=(10, 6))

for target in target_values:
    target_seqs_oh = generate_sequences_in_batches(
        diffusion, real_oracle,
        total_samples=100,
        batch_size=32,
        guidance_scale=2.0,
        target_val=target
    )
    with torch.no_grad():
        actual_scores = real_oracle(target_seqs_oh).cpu().numpy().flatten()
    print(f"  -> Target: {target}, Actual Mean: {np.mean(actual_scores):.4f}")
    sns.kdeplot(actual_scores, label=f'Target {target}', fill=True, alpha=0.3)

plt.title("Conditional Generation: Target vs Actual Score Distribution")
plt.axvline(x=target_values[0], color='r', linestyle='--', alpha=0.5, label='Targets')
for t in target_values[1:]: plt.axvline(x=t, color='r', linestyle='--')
plt.legend()
plt.savefig(os.path.join(output_dir, "target_generation_verification.png"), dpi=300)
plt.show()

print("\nAll tasks completed successfully!")