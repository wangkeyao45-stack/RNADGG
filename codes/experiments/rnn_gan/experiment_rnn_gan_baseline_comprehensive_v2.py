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
from scipy.spatial.distance import jensenshannon
from scipy.linalg import sqrtm
from collections import Counter
import logomaker
from tqdm import tqdm
import math
import Levenshtein


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
output_dir = "output_plots_comprehensive_v22"
os.makedirs(output_dir, exist_ok=True)
print(f"所有图表将被保存在 '{output_dir}/' 文件夹中。")


def plot_logo(one_hot_seqs, title=''):
    """
    使用logomaker库绘制专业的序列标识图
    """
    if isinstance(one_hot_seqs, torch.Tensor):
        one_hot_seqs = one_hot_seqs.detach().cpu().numpy()

    # 确保形状为 (Length, 4)
    if one_hot_seqs.shape[-1] != 4:
        one_hot_seqs = np.transpose(one_hot_seqs, (0, 2, 1))

    # 计算PWM (Position Weight Matrix)
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
    logo.ax.set_ylabel("Probability", labelpad=2, fontsize=10)
    logo.ax.set_xlabel("Position", fontsize=10)
    logo.ax.set_xticks(range(seq_length))
    logo.ax.set_xticklabels(range(1, seq_length + 1))
    logo.ax.spines['left'].set_linewidth(1.5)
    logo.ax.spines['bottom'].set_linewidth(1.5)
    logo.ax.grid(False)
    logo.ax.set_ylim([0, 1])  # 概率总和为1

    if title:
        logo.ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()
    sanitized_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace("/", "")
    if not sanitized_title: sanitized_title = "sequence_logo"

    plt.savefig(os.path.join(output_dir, f"logo_{sanitized_title}.png"), dpi=300, bbox_inches='tight')
    plt.show()


# ==============================================================================
# 第1部分: PyTorch 模型定义 (UNet, Diffusion, Oracle)
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
# 第2部分: 数据加载
# ==============================================================================
print("--- Part 2: Loading Data ---")
if not os.path.exists(CSV_FILE):
    print(f"Warning: '{CSV_FILE}' not found. Generating dummy data for demonstration.")
    # 生成假数据以防没有CSV文件也能运行
    dummy_seqs = ["".join(np.random.choice(['A', 'C', 'G', 'T'], SEQUENCE_LENGTH)) for _ in range(5000)]
    dummy_scores = np.random.rand(5000).astype(np.float32)
    df = pd.DataFrame({'序列': dummy_seqs, 'rl': dummy_scores})
else:
    df = pd.read_csv(CSV_FILE)

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


NUM_SAMPLES_TO_USE = 280000
df_subset = df.head(NUM_SAMPLES_TO_USE)
sequences = df_subset['序列'].tolist()
rl_scores = df_subset['rl'].values.astype(np.float32)
processed_data = [(one_hot_encode(s), r, s) for s, r in zip(sequences, rl_scores) if one_hot_encode(s) is not None]
one_hot_sequences = np.array([item[0] for item in processed_data])
scores = np.array([item[1] for item in processed_data])
real_seq_strings = [item[2] for item in processed_data]
print(f"Data loaded. Shape: {one_hot_sequences.shape}")

# ==============================================================================
# 第3部分: 训练 Oracle
# ==============================================================================
print("\n--- Part 3: Training Real Oracle ---")
X_train, X_val, y_train, y_val = train_test_split(one_hot_sequences, scores, test_size=0.2, random_state=42)
oracle_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32,
                                 shuffle=True)
real_oracle = OracleCNN().to(device)
oracle_optimizer, criterion = optim.Adam(real_oracle.parameters(), lr=1e-3), nn.MSELoss()
ORACLE_EPOCHS = 30  # 稍微减少epoch以加快演示速度，实际使用建议50
for epoch in range(ORACLE_EPOCHS):
    real_oracle.train()
    for seqs, labels in oracle_train_loader:
        oracle_optimizer.zero_grad()
        loss = criterion(real_oracle(seqs.to(device)).squeeze(), labels.to(device))
        loss.backward()
        oracle_optimizer.step()
    if (epoch + 1) % 10 == 0: print(f"Oracle Epoch {epoch + 1}/{ORACLE_EPOCHS} done.")
real_oracle.eval()

# ==============================================================================
# 第4部分: 训练 Diffusion
# ==============================================================================
print("\n--- Part 4: Training Diffusion ---")
diffusion_train_loader = DataLoader(TensorDataset(torch.from_numpy(one_hot_sequences)), batch_size=64, shuffle=True)
unet = UNet().to(device)
diffusion = Diffusion(unet)
diffusion_optimizer = optim.Adam(unet.parameters(), lr=1e-4)
DIFFUSION_EPOCHS = 50  # 演示用50，实际建议100+
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
# 第5部分: 采样与生成函数
# ==============================================================================
print("\n--- Part 5: Generation Functions ---")


def guided_sampling(diffusion_model, oracle_model, batch_size, guidance_scale=0.01, track_gradients=False):
    x_t = torch.randn((batch_size, NUCLEOTIDES, SEQUENCE_LENGTH), device=device)
    gradient_logs = []
    for t in tqdm(reversed(range(diffusion_model.T)), desc="Sampling", total=diffusion_model.T, leave=False):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        if guidance_scale != 0:
            with torch.enable_grad():
                x_t.requires_grad_()
                scores = oracle_model(x_t).sum()
                grad = torch.autograd.grad(scores, x_t)[0]
                if track_gradients: gradient_logs.append(grad.norm().item() / batch_size)
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
                x_t = x_t_minus_1_mean + torch.sqrt(diffusion_model.betas[t]) * torch.randn_like(x_t)
            else:
                x_t = x_t_minus_1_mean
    final_seqs = F.one_hot(torch.argmax(x_t, dim=1), num_classes=NUCLEOTIDES).float().permute(0, 2, 1)
    if track_gradients: return final_seqs, gradient_logs
    return final_seqs


def generate_sequences_in_batches(diffusion_model, oracle_model, total_samples, batch_size, guidance_scale=0.0):
    generated_seqs_list = []
    num_batches = math.ceil(total_samples / batch_size)
    for i in range(num_batches):
        current_batch_size = min(batch_size, total_samples - sum(len(b) for b in generated_seqs_list))
        if current_batch_size <= 0: break
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
    print(f"Saved to {filepath}")


# ==============================================================================
# 第6部分: 传统优化算法 (遗传算法) - [NEW]
# ==============================================================================
class GeneticAlgorithm:
    def __init__(self, oracle_model, seq_len=17, pop_size=200, mutation_rate=0.1):
        self.oracle = oracle_model
        self.seq_len = seq_len
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.n_nucleotides = 4

    def initial_population(self):
        # 随机初始化种群 (One-hot)
        indices = torch.randint(0, self.n_nucleotides, (self.pop_size, self.seq_len), device=device)
        return F.one_hot(indices, num_classes=self.n_nucleotides).float().permute(0, 2, 1)

    def optimize(self, generations=50):
        population = self.initial_population()
        for g in tqdm(range(generations), desc="Running GA", leave=False):
            with torch.no_grad():
                scores = self.oracle(population).flatten()

            # Elitism: 保留前20%
            sorted_idx = torch.argsort(scores, descending=True)
            num_elites = int(self.pop_size * 0.2)
            elites = population[sorted_idx[:num_elites]]

            # Parents: 简单地从上半区随机选
            parents = population[sorted_idx[:self.pop_size // 2]]

            # Crossover
            idx_1 = torch.randint(0, len(parents), (self.pop_size - num_elites,), device=device)
            idx_2 = torch.randint(0, len(parents), (self.pop_size - num_elites,), device=device)
            p1, p2 = parents[idx_1], parents[idx_2]

            crossover_pt = torch.randint(1, self.seq_len, (self.pop_size - num_elites, 1, 1), device=device)
            mask = torch.arange(self.seq_len, device=device).view(1, 1, -1) < crossover_pt
            offspring = torch.where(mask, p1, p2)

            # Mutation
            mutation_mask = torch.rand(offspring.shape[0], self.seq_len, device=device) < self.mutation_rate
            if mutation_mask.any():
                random_indices = torch.randint(0, self.n_nucleotides, (offspring.shape[0], self.seq_len), device=device)
                random_onehot = F.one_hot(random_indices, num_classes=self.n_nucleotides).float().permute(0, 2, 1)
                mask_expanded = mutation_mask.unsqueeze(1).expand_as(offspring)
                offspring = torch.where(mask_expanded, random_onehot, offspring)

            population = torch.cat([elites, offspring], dim=0)
        return population


# ==============================================================================
# 第7部分: 综合评估 (对比 传统方法, 基线, RL-Diffusion)
# ==============================================================================
print("\n--- Part 7: Comprehensive Evaluation (Enhanced) ---")


# 1. 辅助函数
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
    return [], torch.tensor([]).to(device)


def get_kmer_dist(seqs, k=4):
    cnt = Counter([s[i:i + k] for s in seqs for i in range(len(s) - k + 1)])
    total = sum(cnt.values())
    return {k: v / total for k, v in cnt.items()}


# 2. 生成评估数据
N_EVAL = 200
BATCH_SIZE = 32
GUIDANCE = 0.5

print(f"Generating data for comparison (N={N_EVAL})...")

# A. Real Data (随机抽取真实数据作为基准)
real_idx = np.random.choice(len(one_hot_sequences), N_EVAL, replace=False)
real_oh_eval = torch.from_numpy(one_hot_sequences[real_idx]).to(device)
real_str_eval = [real_seq_strings[i] for i in real_idx]

# B. Random
print("-> Generating Random...")
rand_str, rand_oh = generate_random_sequences(N_EVAL, SEQUENCE_LENGTH)

# C. Baseline Diffusion
print("-> Generating Baseline Diffusion...")
base_oh = generate_sequences_in_batches(diffusion, real_oracle, N_EVAL, BATCH_SIZE, 0.0)
base_str = one_hot_to_strings(base_oh)

# D. RL-Guided Diffusion
print("-> Generating RL-Guided Diffusion...")
rl_oh = generate_sequences_in_batches(diffusion, real_oracle, N_EVAL, BATCH_SIZE, GUIDANCE)
rl_str = one_hot_to_strings(rl_oh)

# E. Traditional GA [NEW]
print("-> Running Genetic Algorithm...")
ga_solver = GeneticAlgorithm(real_oracle, seq_len=SEQUENCE_LENGTH, pop_size=N_EVAL)
ga_oh = ga_solver.optimize(generations=50)
ga_str = one_hot_to_strings(ga_oh)

# 3. 计算指标
print("\n--- Calculating Metrics ---")
# 准备JSD基准分布
real_full_dist = get_kmer_dist(real_seq_strings)
all_k = sorted(list(real_full_dist.keys()))


def calc_lev(gen_str):
    # 计算与真实数据集的平均最小距离（多样性/拟合度指标）
    # 为了速度，只与部分真实数据比较
    subset_real = real_str_eval
    return sum([min([Levenshtein.distance(g, r) for r in subset_real]) for g in gen_str]) / len(gen_str)


def calc_jsd_val(gen_str):
    d = get_kmer_dist(gen_str)
    p = np.array([real_full_dist.get(k, 0) for k in all_k])
    q = np.array([d.get(k, 0) for k in all_k])
    return jensenshannon(p, q, base=2.0)


def get_embs_safe(data):
    embs = []
    with torch.no_grad():
        for i in range(0, data.shape[0], 32):
            batch = data[i:i + 32]
            batch_emb = real_oracle.get_embedding(batch)
            embs.append(batch_emb.detach().cpu().numpy())
    return np.concatenate(embs, axis=0)


def calc_fid_val(gen_oh):
    real_embs = get_embs_safe(real_oh_eval)
    gen_embs = get_embs_safe(gen_oh)
    mu1, sig1 = np.mean(real_embs, axis=0), np.cov(real_embs, rowvar=False)
    mu2, sig2 = np.mean(gen_embs, axis=0), np.cov(gen_embs, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sig1.dot(sig2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return diff.dot(diff) + np.trace(sig1 + sig2 - 2 * covmean)


# 收集结果
metrics = []
datasets = [
    ('Random', rand_str, rand_oh),
    ('Baseline Diff', base_str, base_oh),
    ('Traditional GA', ga_str, ga_oh),
    ('RL-Guided Diff', rl_str, rl_oh)
]

print(f"{'Method':<20} | {'Lev Dist':<10} | {'JSD':<10} | {'FID':<10}")
print("-" * 60)
for name, s_str, s_oh in datasets:
    lev = calc_lev(s_str)
    jsd = calc_jsd_val(s_str)
    fid = calc_fid_val(s_oh)
    print(f"{name:<20} | {lev:<10.2f} | {jsd:<10.4f} | {fid:<10.4f}")

# 4. 绘图: 奖励分布对比
print("\n--- Plotting Reward Distribution ---")
with torch.no_grad():
    s_real = real_oracle(real_oh_eval).cpu().numpy().flatten()
    s_rand = real_oracle(rand_oh).cpu().numpy().flatten()
    s_base = real_oracle(base_oh).cpu().numpy().flatten()
    s_rl = real_oracle(rl_oh).cpu().numpy().flatten()
    s_ga = real_oracle(ga_oh).cpu().numpy().flatten()

df_viol = pd.DataFrame({
    'Method': ['Real'] * N_EVAL + ['Random'] * N_EVAL + ['Baseline'] * N_EVAL + ['RL-Guided'] * N_EVAL + [
        'Traditional GA'] * N_EVAL,
    'Score': np.concatenate([s_real, s_rand, s_base, s_rl, s_ga])
})

plt.figure(figsize=(12, 6))
sns.violinplot(x='Method', y='Score', data=df_viol, palette="Set3", hue='Method', legend=False)
plt.title("Optimization Performance: Deep Learning vs Traditional GA")
plt.ylabel("Predicted Activity Score")
plt.savefig(os.path.join(output_dir, "comparison_violin.png"), dpi=300)
plt.show()

# 5. 绘图: Global Sequence Logos (200条)
print("\n--- Plotting Global Sequence Logos (Consensus) ---")
# 绘制所有组的Logo
plot_data = [
    ("Real Data (Subset)", real_oh_eval),
    ("Random Baseline", rand_oh),
    ("Diffusion Baseline", base_oh),
    ("Traditional GA", ga_oh),
    ("RL-Guided Diffusion", rl_oh)
]

for title, tensor_data in plot_data:
    print(f"Generating logo for: {title}")
    plot_logo(tensor_data, title=f"{title} (n={N_EVAL})")

# ==============================================================================
# 第8部分: 梯度分析与保存
# ==============================================================================
print("\n--- Part 8: Gradient Analysis & Saving ---")

# 梯度动态
_, grad_logs = guided_sampling(diffusion, real_oracle, batch_size=16, guidance_scale=GUIDANCE, track_gradients=True)
plt.figure(figsize=(10, 5))
plt.plot(grad_logs, color='purple')
plt.xlabel('Denoising Steps (T -> 0)')
plt.ylabel('Gradient Norm')
plt.title('Guidance Strength during Generation')
plt.savefig(os.path.join(output_dir, "gradient_dynamics.png"), dpi=300)
plt.show()

# 保存序列
save_sequences_to_txt("final_seqs_baseline.txt", base_str, s_base)
save_sequences_to_txt("final_seqs_ga.txt", ga_str, s_ga)
save_sequences_to_txt("final_seqs_rl_guided.txt", rl_str, s_rl)

print("\nProcessing Complete! Check the output folder for all plots and files.")


def plot_top_k_logos(methods_data, k=5):
    """
    methods_data: List of (name, oh_tensor, scores)
    """
    num_methods = len(methods_data)
    fig, axes = plt.subplots(num_methods, 1, figsize=(10, 2.5 * num_methods))

    for i, (name, oh, scores) in enumerate(methods_data):
        # 获取 top k 索引
        top_idx = np.argsort(scores)[-k:]
        top_oh = oh[top_idx]

        # 计算 PWM
        pwm = torch.mean(top_oh, dim=0).cpu().numpy()
        pwm_df = pd.DataFrame(pwm.T, columns=['A', 'C', 'G', 'T'])

        ax = axes[i] if num_methods > 1 else axes
        logo = logomaker.Logo(pwm_df, ax=ax, color_scheme='colorblind_safe')
        ax.set_title(f"Top {k} Sequences: {name}", fontsize=12)
        ax.set_ylabel("Prob")
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_5_comparison.png"), dpi=300)
    plt.show()


# 调用示例 (在 Part 7 后面执行):
comparison_data = [
    ('Random', rand_oh, s_rand),
    ('GA', ga_oh, s_ga),
    ('Baseline Diff', base_oh, s_base),
    ('RL-Guided Diff', rl_oh, s_rl)
]
plot_top_k_logos(comparison_data)

from sklearn.manifold import TSNE


def plot_tsne_distribution(methods_data, real_oh):
    """
    可视化不同方法生成的序列在特征空间中的分布
    """
    all_embs = []
    labels = []

    # 获取真实数据作为基准
    real_emb = get_embs_safe(real_oh)
    all_embs.append(real_emb)
    labels.extend(['Real'] * len(real_emb))

    for name, oh, _ in methods_data:
        emb = get_embs_safe(oh)
        all_embs.append(emb)
        labels.extend([name] * len(emb))

    X = np.concatenate(all_embs, axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(X_embedded, columns=['tsne_1', 'tsne_2'])
    df_tsne['Method'] = labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='tsne_1', y='tsne_2', hue='Method', alpha=0.6, s=40, palette='viridis')
    plt.title("Sequence Space Distribution (t-SNE)")
    plt.savefig(os.path.join(output_dir, "tsne_space.png"), dpi=300)
    plt.show()


# 调用
plot_tsne_distribution(comparison_data, real_oh_eval)


def analyze_controllability(diffusion_model, oracle_model, scales=[0.0, 0.1, 0.5, 1.0]):
    """
    测试不同的引导强度对得分分布的影响
    """
    results = []
    for s in scales:
        print(f"Testing Guidance Scale: {s}")
        gen_oh = generate_sequences_in_batches(diffusion_model, oracle_model, 100, 32, guidance_scale=s)
        with torch.no_grad():
            gen_scores = oracle_model(gen_oh).cpu().numpy().flatten()

        tmp_df = pd.DataFrame({'Score': gen_scores, 'Guidance': [str(s)] * len(gen_scores)})
        results.append(tmp_df)

    df_res = pd.concat(results)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_res, x='Score', hue='Guidance', fill=True, common_norm=False)
    plt.axvline(x=scores.max(), color='red', linestyle='--', label='Max in Training Data')
    plt.title("Controllable Generation: Score Distribution vs Guidance Strength")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "control_performance.png"), dpi=300)
    plt.show()


# 调用
analyze_controllability(diffusion, real_oracle)# -*- coding: utf-8 -*-
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
from scipy.spatial.distance import jensenshannon
from scipy.linalg import sqrtm
from collections import Counter
import logomaker
from tqdm import tqdm
import math
import Levenshtein


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
output_dir = "output_plots_comprehensive_v2"
os.makedirs(output_dir, exist_ok=True)
print(f"所有图表将被保存在 '{output_dir}/' 文件夹中。")


def plot_logo(one_hot_seqs, title=''):
    """
    使用logomaker库绘制专业的序列标识图
    """
    if isinstance(one_hot_seqs, torch.Tensor):
        one_hot_seqs = one_hot_seqs.detach().cpu().numpy()

    # 确保形状为 (Length, 4)
    if one_hot_seqs.shape[-1] != 4:
        one_hot_seqs = np.transpose(one_hot_seqs, (0, 2, 1))

    # 计算PWM (Position Weight Matrix)
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
    logo.ax.set_ylabel("Probability", labelpad=2, fontsize=10)
    logo.ax.set_xlabel("Position", fontsize=10)
    logo.ax.set_xticks(range(seq_length))
    logo.ax.set_xticklabels(range(1, seq_length + 1))
    logo.ax.spines['left'].set_linewidth(1.5)
    logo.ax.spines['bottom'].set_linewidth(1.5)
    logo.ax.grid(False)
    logo.ax.set_ylim([0, 1])  # 概率总和为1

    if title:
        logo.ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()
    sanitized_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace("/", "")
    if not sanitized_title: sanitized_title = "sequence_logo"

    plt.savefig(os.path.join(output_dir, f"logo_{sanitized_title}.png"), dpi=300, bbox_inches='tight')
    plt.show()


# ==============================================================================
# 第1部分: PyTorch 模型定义 (UNet, Diffusion, Oracle)
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
# 第2部分: 数据加载
# ==============================================================================
print("--- Part 2: Loading Data ---")
if not os.path.exists(CSV_FILE):
    print(f"Warning: '{CSV_FILE}' not found. Generating dummy data for demonstration.")
    # 生成假数据以防没有CSV文件也能运行
    dummy_seqs = ["".join(np.random.choice(['A', 'C', 'G', 'T'], SEQUENCE_LENGTH)) for _ in range(5000)]
    dummy_scores = np.random.rand(5000).astype(np.float32)
    df = pd.DataFrame({'序列': dummy_seqs, 'rl': dummy_scores})
else:
    df = pd.read_csv(CSV_FILE)

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


NUM_SAMPLES_TO_USE = 280000
df_subset = df.head(NUM_SAMPLES_TO_USE)
sequences = df_subset['序列'].tolist()
rl_scores = df_subset['rl'].values.astype(np.float32)
processed_data = [(one_hot_encode(s), r, s) for s, r in zip(sequences, rl_scores) if one_hot_encode(s) is not None]
one_hot_sequences = np.array([item[0] for item in processed_data])
scores = np.array([item[1] for item in processed_data])
real_seq_strings = [item[2] for item in processed_data]
print(f"Data loaded. Shape: {one_hot_sequences.shape}")

# ==============================================================================
# 第3部分: 训练 Oracle
# ==============================================================================
print("\n--- Part 3: Training Real Oracle ---")
X_train, X_val, y_train, y_val = train_test_split(one_hot_sequences, scores, test_size=0.2, random_state=42)
oracle_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32,
                                 shuffle=True)
real_oracle = OracleCNN().to(device)
oracle_optimizer, criterion = optim.Adam(real_oracle.parameters(), lr=1e-3), nn.MSELoss()
ORACLE_EPOCHS = 30  # 稍微减少epoch以加快演示速度，实际使用建议50
for epoch in range(ORACLE_EPOCHS):
    real_oracle.train()
    for seqs, labels in oracle_train_loader:
        oracle_optimizer.zero_grad()
        loss = criterion(real_oracle(seqs.to(device)).squeeze(), labels.to(device))
        loss.backward()
        oracle_optimizer.step()
    if (epoch + 1) % 10 == 0: print(f"Oracle Epoch {epoch + 1}/{ORACLE_EPOCHS} done.")
real_oracle.eval()

# ==============================================================================
# 第4部分: 训练 Diffusion
# ==============================================================================
print("\n--- Part 4: Training Diffusion ---")
diffusion_train_loader = DataLoader(TensorDataset(torch.from_numpy(one_hot_sequences)), batch_size=64, shuffle=True)
unet = UNet().to(device)
diffusion = Diffusion(unet)
diffusion_optimizer = optim.Adam(unet.parameters(), lr=1e-4)
DIFFUSION_EPOCHS = 50  # 演示用50，实际建议100+
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
# 第5部分: 采样与生成函数
# ==============================================================================
print("\n--- Part 5: Generation Functions ---")


def guided_sampling(diffusion_model, oracle_model, batch_size, guidance_scale=0.01, track_gradients=False):
    x_t = torch.randn((batch_size, NUCLEOTIDES, SEQUENCE_LENGTH), device=device)
    gradient_logs = []
    for t in tqdm(reversed(range(diffusion_model.T)), desc="Sampling", total=diffusion_model.T, leave=False):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        if guidance_scale != 0:
            with torch.enable_grad():
                x_t.requires_grad_()
                scores = oracle_model(x_t).sum()
                grad = torch.autograd.grad(scores, x_t)[0]
                if track_gradients: gradient_logs.append(grad.norm().item() / batch_size)
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
                x_t = x_t_minus_1_mean + torch.sqrt(diffusion_model.betas[t]) * torch.randn_like(x_t)
            else:
                x_t = x_t_minus_1_mean
    final_seqs = F.one_hot(torch.argmax(x_t, dim=1), num_classes=NUCLEOTIDES).float().permute(0, 2, 1)
    if track_gradients: return final_seqs, gradient_logs
    return final_seqs


def generate_sequences_in_batches(diffusion_model, oracle_model, total_samples, batch_size, guidance_scale=0.0):
    generated_seqs_list = []
    num_batches = math.ceil(total_samples / batch_size)
    for i in range(num_batches):
        current_batch_size = min(batch_size, total_samples - sum(len(b) for b in generated_seqs_list))
        if current_batch_size <= 0: break
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
    print(f"Saved to {filepath}")


# ==============================================================================
# 第6部分: 传统优化算法 (遗传算法) - [NEW]
# ==============================================================================
class GeneticAlgorithm:
    def __init__(self, oracle_model, seq_len=17, pop_size=200, mutation_rate=0.1):
        self.oracle = oracle_model
        self.seq_len = seq_len
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.n_nucleotides = 4

    def initial_population(self):
        # 随机初始化种群 (One-hot)
        indices = torch.randint(0, self.n_nucleotides, (self.pop_size, self.seq_len), device=device)
        return F.one_hot(indices, num_classes=self.n_nucleotides).float().permute(0, 2, 1)

    def optimize(self, generations=50):
        population = self.initial_population()
        for g in tqdm(range(generations), desc="Running GA", leave=False):
            with torch.no_grad():
                scores = self.oracle(population).flatten()

            # Elitism: 保留前20%
            sorted_idx = torch.argsort(scores, descending=True)
            num_elites = int(self.pop_size * 0.2)
            elites = population[sorted_idx[:num_elites]]

            # Parents: 简单地从上半区随机选
            parents = population[sorted_idx[:self.pop_size // 2]]

            # Crossover
            idx_1 = torch.randint(0, len(parents), (self.pop_size - num_elites,), device=device)
            idx_2 = torch.randint(0, len(parents), (self.pop_size - num_elites,), device=device)
            p1, p2 = parents[idx_1], parents[idx_2]

            crossover_pt = torch.randint(1, self.seq_len, (self.pop_size - num_elites, 1, 1), device=device)
            mask = torch.arange(self.seq_len, device=device).view(1, 1, -1) < crossover_pt
            offspring = torch.where(mask, p1, p2)

            # Mutation
            mutation_mask = torch.rand(offspring.shape[0], self.seq_len, device=device) < self.mutation_rate
            if mutation_mask.any():
                random_indices = torch.randint(0, self.n_nucleotides, (offspring.shape[0], self.seq_len), device=device)
                random_onehot = F.one_hot(random_indices, num_classes=self.n_nucleotides).float().permute(0, 2, 1)
                mask_expanded = mutation_mask.unsqueeze(1).expand_as(offspring)
                offspring = torch.where(mask_expanded, random_onehot, offspring)

            population = torch.cat([elites, offspring], dim=0)
        return population


# ==============================================================================
# 第7部分: 综合评估 (对比 传统方法, 基线, RL-Diffusion)
# ==============================================================================
print("\n--- Part 7: Comprehensive Evaluation (Enhanced) ---")


# 1. 辅助函数
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
    return [], torch.tensor([]).to(device)


def get_kmer_dist(seqs, k=4):
    cnt = Counter([s[i:i + k] for s in seqs for i in range(len(s) - k + 1)])
    total = sum(cnt.values())
    return {k: v / total for k, v in cnt.items()}


# 2. 生成评估数据
N_EVAL = 200
BATCH_SIZE = 32
GUIDANCE = 0.5

print(f"Generating data for comparison (N={N_EVAL})...")

# A. Real Data (随机抽取真实数据作为基准)
real_idx = np.random.choice(len(one_hot_sequences), N_EVAL, replace=False)
real_oh_eval = torch.from_numpy(one_hot_sequences[real_idx]).to(device)
real_str_eval = [real_seq_strings[i] for i in real_idx]

# B. Random
print("-> Generating Random...")
rand_str, rand_oh = generate_random_sequences(N_EVAL, SEQUENCE_LENGTH)

# C. Baseline Diffusion
print("-> Generating Baseline Diffusion...")
base_oh = generate_sequences_in_batches(diffusion, real_oracle, N_EVAL, BATCH_SIZE, 0.0)
base_str = one_hot_to_strings(base_oh)

# D. RL-Guided Diffusion
print("-> Generating RL-Guided Diffusion...")
rl_oh = generate_sequences_in_batches(diffusion, real_oracle, N_EVAL, BATCH_SIZE, GUIDANCE)
rl_str = one_hot_to_strings(rl_oh)

# E. Traditional GA [NEW]
print("-> Running Genetic Algorithm...")
ga_solver = GeneticAlgorithm(real_oracle, seq_len=SEQUENCE_LENGTH, pop_size=N_EVAL)
ga_oh = ga_solver.optimize(generations=50)
ga_str = one_hot_to_strings(ga_oh)

# 3. 计算指标
print("\n--- Calculating Metrics ---")
# 准备JSD基准分布
real_full_dist = get_kmer_dist(real_seq_strings)
all_k = sorted(list(real_full_dist.keys()))


def calc_lev(gen_str):
    # 计算与真实数据集的平均最小距离（多样性/拟合度指标）
    # 为了速度，只与部分真实数据比较
    subset_real = real_str_eval
    return sum([min([Levenshtein.distance(g, r) for r in subset_real]) for g in gen_str]) / len(gen_str)


def calc_jsd_val(gen_str):
    d = get_kmer_dist(gen_str)
    p = np.array([real_full_dist.get(k, 0) for k in all_k])
    q = np.array([d.get(k, 0) for k in all_k])
    return jensenshannon(p, q, base=2.0)


def get_embs_safe(data):
    embs = []
    with torch.no_grad():
        for i in range(0, data.shape[0], 32):
            batch = data[i:i + 32]
            batch_emb = real_oracle.get_embedding(batch)
            embs.append(batch_emb.detach().cpu().numpy())
    return np.concatenate(embs, axis=0)


def calc_fid_val(gen_oh):
    real_embs = get_embs_safe(real_oh_eval)
    gen_embs = get_embs_safe(gen_oh)
    mu1, sig1 = np.mean(real_embs, axis=0), np.cov(real_embs, rowvar=False)
    mu2, sig2 = np.mean(gen_embs, axis=0), np.cov(gen_embs, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sig1.dot(sig2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return diff.dot(diff) + np.trace(sig1 + sig2 - 2 * covmean)


# 收集结果
metrics = []
datasets = [
    ('Random', rand_str, rand_oh),
    ('Baseline Diff', base_str, base_oh),
    ('Traditional GA', ga_str, ga_oh),
    ('RL-Guided Diff', rl_str, rl_oh)
]

print(f"{'Method':<20} | {'Lev Dist':<10} | {'JSD':<10} | {'FID':<10}")
print("-" * 60)
for name, s_str, s_oh in datasets:
    lev = calc_lev(s_str)
    jsd = calc_jsd_val(s_str)
    fid = calc_fid_val(s_oh)
    print(f"{name:<20} | {lev:<10.2f} | {jsd:<10.4f} | {fid:<10.4f}")

# 4. 绘图: 奖励分布对比
print("\n--- Plotting Reward Distribution ---")
with torch.no_grad():
    s_real = real_oracle(real_oh_eval).cpu().numpy().flatten()
    s_rand = real_oracle(rand_oh).cpu().numpy().flatten()
    s_base = real_oracle(base_oh).cpu().numpy().flatten()
    s_rl = real_oracle(rl_oh).cpu().numpy().flatten()
    s_ga = real_oracle(ga_oh).cpu().numpy().flatten()

df_viol = pd.DataFrame({
    'Method': ['Real'] * N_EVAL + ['Random'] * N_EVAL + ['Baseline'] * N_EVAL + ['RL-Guided'] * N_EVAL + [
        'Traditional GA'] * N_EVAL,
    'Score': np.concatenate([s_real, s_rand, s_base, s_rl, s_ga])
})

plt.figure(figsize=(12, 6))
sns.violinplot(x='Method', y='Score', data=df_viol, palette="Set3", hue='Method', legend=False)
plt.title("Optimization Performance: Deep Learning vs Traditional GA")
plt.ylabel("Predicted Activity Score")
plt.savefig(os.path.join(output_dir, "comparison_violin.png"), dpi=300)
plt.show()

# 5. 绘图: Global Sequence Logos (200条)
print("\n--- Plotting Global Sequence Logos (Consensus) ---")
# 绘制所有组的Logo
plot_data = [
    ("Real Data (Subset)", real_oh_eval),
    ("Random Baseline", rand_oh),
    ("Diffusion Baseline", base_oh),
    ("Traditional GA", ga_oh),
    ("RL-Guided Diffusion", rl_oh)
]

for title, tensor_data in plot_data:
    print(f"Generating logo for: {title}")
    plot_logo(tensor_data, title=f"{title} (n={N_EVAL})")

# ==============================================================================
# 第8部分: 梯度分析与保存
# ==============================================================================
print("\n--- Part 8: Gradient Analysis & Saving ---")

# 梯度动态
_, grad_logs = guided_sampling(diffusion, real_oracle, batch_size=16, guidance_scale=GUIDANCE, track_gradients=True)
plt.figure(figsize=(10, 5))
plt.plot(grad_logs, color='purple')
plt.xlabel('Denoising Steps (T -> 0)')
plt.ylabel('Gradient Norm')
plt.title('Guidance Strength during Generation')
plt.savefig(os.path.join(output_dir, "gradient_dynamics.png"), dpi=300)
plt.show()

# 保存序列
save_sequences_to_txt("final_seqs_baseline.txt", base_str, s_base)
save_sequences_to_txt("final_seqs_ga.txt", ga_str, s_ga)
save_sequences_to_txt("final_seqs_rl_guided.txt", rl_str, s_rl)

print("\nProcessing Complete! Check the output folder for all plots and files.")


def plot_top_k_logos(methods_data, k=5):
    """
    methods_data: List of (name, oh_tensor, scores)
    """
    num_methods = len(methods_data)
    fig, axes = plt.subplots(num_methods, 1, figsize=(10, 2.5 * num_methods))

    for i, (name, oh, scores) in enumerate(methods_data):
        # 获取 top k 索引
        top_idx = np.argsort(scores)[-k:]
        top_oh = oh[top_idx]

        # 计算 PWM
        pwm = torch.mean(top_oh, dim=0).cpu().numpy()
        pwm_df = pd.DataFrame(pwm.T, columns=['A', 'C', 'G', 'T'])

        ax = axes[i] if num_methods > 1 else axes
        logo = logomaker.Logo(pwm_df, ax=ax, color_scheme='colorblind_safe')
        ax.set_title(f"Top {k} Sequences: {name}", fontsize=12)
        ax.set_ylabel("Prob")
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_5_comparison.png"), dpi=300)
    plt.show()


# 调用示例 (在 Part 7 后面执行):
comparison_data = [
    ('Random', rand_oh, s_rand),
    ('GA', ga_oh, s_ga),
    ('Baseline Diff', base_oh, s_base),
    ('RL-Guided Diff', rl_oh, s_rl)
]
plot_top_k_logos(comparison_data)

from sklearn.manifold import TSNE


def plot_tsne_distribution(methods_data, real_oh):
    """
    可视化不同方法生成的序列在特征空间中的分布
    """
    all_embs = []
    labels = []

    # 获取真实数据作为基准
    real_emb = get_embs_safe(real_oh)
    all_embs.append(real_emb)
    labels.extend(['Real'] * len(real_emb))

    for name, oh, _ in methods_data:
        emb = get_embs_safe(oh)
        all_embs.append(emb)
        labels.extend([name] * len(emb))

    X = np.concatenate(all_embs, axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(X_embedded, columns=['tsne_1', 'tsne_2'])
    df_tsne['Method'] = labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='tsne_1', y='tsne_2', hue='Method', alpha=0.6, s=40, palette='viridis')
    plt.title("Sequence Space Distribution (t-SNE)")
    plt.savefig(os.path.join(output_dir, "tsne_space.png"), dpi=300)
    plt.show()


# 调用
plot_tsne_distribution(comparison_data, real_oh_eval)


def analyze_controllability(diffusion_model, oracle_model, scales=[0.0, 0.1, 0.5, 1.0]):
    """
    测试不同的引导强度对得分分布的影响
    """
    results = []
    for s in scales:
        print(f"Testing Guidance Scale: {s}")
        gen_oh = generate_sequences_in_batches(diffusion_model, oracle_model, 100, 32, guidance_scale=s)
        with torch.no_grad():
            gen_scores = oracle_model(gen_oh).cpu().numpy().flatten()

        tmp_df = pd.DataFrame({'Score': gen_scores, 'Guidance': [str(s)] * len(gen_scores)})
        results.append(tmp_df)

    df_res = pd.concat(results)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_res, x='Score', hue='Guidance', fill=True, common_norm=False)
    plt.axvline(x=scores.max(), color='red', linestyle='--', label='Max in Training Data')
    plt.title("Controllable Generation: Score Distribution vs Guidance Strength")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "control_performance.png"), dpi=300)
    plt.show()


# 调用
analyze_controllability(diffusion, real_oracle)