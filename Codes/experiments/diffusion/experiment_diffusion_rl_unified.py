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
output_dir = "output_plots_diffusion_finetune"
os.makedirs(output_dir, exist_ok=True)
print(f"所有图表将被保存在 '{output_dir}/' 文件夹中。")




def plot_logo(one_hot_seqs, title=''):
    """
    使用logomaker库绘制专业的序列标识图
    """
    # 数据类型转换和形状调整
    if isinstance(one_hot_seqs, torch.Tensor):
        one_hot_seqs = one_hot_seqs.detach().cpu().numpy()

    if one_hot_seqs.shape[-1] != 4:
        one_hot_seqs = np.transpose(one_hot_seqs, (0, 2, 1))

    # 计算位置频率矩阵(PWM)
    pwm = np.mean(one_hot_seqs, axis=0)

    # 创建PWM的DataFrame，列顺序为A,C,G,T
    pwm_df = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])

    # 设置图形大小，根据序列长度自适应
    seq_length = pwm_df.shape[0]
    fig_width = max(10, seq_length * 0.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 2.5))

    # 设置白色背景
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 使用logomaker创建专业序列标识图
    logo = logomaker.Logo(pwm_df,
                          ax=ax,
                          shade_below=0.5,
                          fade_below=0.5,
                          font_name='sans-serif',
                          color_scheme='colorblind_safe')

    # 样式设置
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)

    # 设置坐标轴标签和样式
    logo.ax.set_ylabel("Information (bits)", labelpad=2, fontsize=10)
    logo.ax.set_xlabel("Position", fontsize=10)

    # 设置x轴刻度
    logo.ax.set_xticks(range(seq_length))
    logo.ax.set_xticklabels(range(1, seq_length + 1))

    # 设置边框线宽
    logo.ax.spines['left'].set_linewidth(1.5)
    logo.ax.spines['bottom'].set_linewidth(1.5)

    # === 新增：去除网格线 ===
    logo.ax.grid(False)  # 关闭所有网格线
    # 或者更具体地只关闭水平网格线：
    # logo.ax.yaxis.grid(False)

    # 设置标题
    if title:
        logo.ax.set_title(title, fontsize=12, pad=10)

    # 调整布局
    plt.tight_layout()

    # 生成安全的文件名
    sanitized_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace("/", "")
    if not sanitized_title:
        sanitized_title = "sequence_logo"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    plt.savefig(os.path.join(output_dir, f"logo_{sanitized_title}.png"),
                dpi=300, bbox_inches='tight', transparent=False)
    plt.show()

# ==============================================================================
# 第1部分: PyTorch 模型定义 (Diffusion)
# ==============================================================================
SEQUENCE_LENGTH = 50
NUCLEOTIDES = 4
CSV_FILE = "processed_data.csv"

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
        self.time_embedding = nn.Sequential(TimeEmbedding(n_channels), nn.Linear(n_channels, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.in_conv = nn.Conv1d(NUCLEOTIDES, n_channels, kernel_size=3, padding=1)
        self.down1 = UpDownBlock(n_channels, n_channels*2, time_emb_dim, down=True)
        self.down2 = UpDownBlock(n_channels*2, n_channels*4, time_emb_dim, down=True)
        self.mid_res1 = ResidualBlock(n_channels*4, n_channels*4, time_emb_dim)
        self.mid_attn = AttentionBlock(n_channels*4)
        self.mid_res2 = ResidualBlock(n_channels*4, n_channels*4, time_emb_dim)
        self.up1 = UpDownBlock(n_channels*8, n_channels*2, time_emb_dim, down=False)
        self.up2 = UpDownBlock(n_channels*4, n_channels, time_emb_dim, down=False)
        self.out_res = ResidualBlock(n_channels*2, n_channels, time_emb_dim)
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
        if x_up.shape[2] != x1.shape[2]: x_up = F.interpolate(x_up, size=x1.shape[2], mode='linear', align_corners=False)
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
# 第2, 3, 4, 5部分: 数据加载和模型训练
# ==============================================================================
print("--- Part 2: Loading and Preprocessing Data from CSV ---")
if not os.path.exists(CSV_FILE): sys.exit(f"错误: 数据文件 '{CSV_FILE}' 不存在。")
char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
idx_to_char = {v: k for k, v in char_map.items()}

def one_hot_encode(seq):
    if len(seq) != SEQUENCE_LENGTH: return None
    encoded = np.zeros((NUCLEOTIDES, SEQUENCE_LENGTH), dtype=np.float32)
    for i, char in enumerate(seq.upper()):
        if char in char_map: encoded[char_map[char], i] = 1.0
        else: return None
    return encoded

df = pd.read_csv(CSV_FILE)
NUM_SAMPLES_TO_USE = 280000
df_subset = df.head(NUM_SAMPLES_TO_USE)
print(f"--- Using a subset of {NUM_SAMPLES_TO_USE} samples. ---")

sequences = df_subset['序列'].tolist()
r1_scores = df_subset['r1'].values.astype(np.float32)
processed_data = [(one_hot_encode(s), r, s) for s, r in zip(sequences, r1_scores) if one_hot_encode(s) is not None]
one_hot_sequences = np.array([item[0] for item in processed_data])
scores = np.array([item[1] for item in processed_data])
real_seq_strings = [item[2] for item in processed_data]
print(f"Data loaded and encoded. Final Shape: {one_hot_sequences.shape}")

print("\n--- Part 3: Training the Real Oracle Model ---")
X_train, X_val, y_train, y_val = train_test_split(one_hot_sequences, scores, test_size=0.2, random_state=42)
oracle_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32, shuffle=True)
oracle_val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=32)
real_oracle = OracleCNN().to(device)
oracle_optimizer, criterion = optim.Adam(real_oracle.parameters(), lr=1e-3), nn.MSELoss()
ORACLE_EPOCHS = 20
for epoch in range(ORACLE_EPOCHS):
    real_oracle.train()
    train_loss_epoch = 0
    for seqs, labels in oracle_train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        oracle_optimizer.zero_grad()
        outputs = real_oracle(seqs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        oracle_optimizer.step()
        train_loss_epoch += loss.item()
    real_oracle.eval()
    val_loss = 0
    with torch.no_grad():
        for seqs, labels in oracle_val_loader:
            outputs = real_oracle(seqs.to(device)).squeeze()
            val_loss += criterion(outputs, labels.to(device)).item()
    if (epoch + 1) % 5 == 0:
        print(
            f"Oracle Epoch [{epoch + 1}/{ORACLE_EPOCHS}], Train Loss: {train_loss_epoch / len(oracle_train_loader):.4f}, Val Loss: {val_loss / len(oracle_val_loader):.4f}")
print("--- Real Oracle training finished. ---")
real_oracle.eval()
torch.cuda.empty_cache()

print("\n--- Part 4: Training the Diffusion Model ---")
# --- MODIFICATION 2: 降低训练批次大小 ---
diffusion_train_loader = DataLoader(TensorDataset(torch.from_numpy(one_hot_sequences)), batch_size=64, shuffle=True)
unet = UNet().to(device)
diffusion = Diffusion(unet)
diffusion_optimizer = optim.Adam(unet.parameters(), lr=1e-4)
DIFFUSION_EPOCHS = 100
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
torch.cuda.empty_cache()  # --- MODIFICATION 4: 清理显存 ---


print("\n--- Part 5: RL-Guided Generative Process ---")
def guided_sampling(diffusion_model, oracle_model, batch_size, guidance_scale=0.01):
    x_t = torch.randn((batch_size, NUCLEOTIDES, SEQUENCE_LENGTH), device=device)
    for t in tqdm(reversed(range(diffusion_model.T)), desc="Guided Sampling", total=diffusion_model.T):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        with torch.enable_grad():
            x_t.requires_grad_()
            scores = oracle_model(x_t).sum()
            grad = torch.autograd.grad(scores, x_t)[0].clamp(-1,1)
        predicted_noise = diffusion_model.model(x_t.detach(), t_tensor.float())
        alpha_t, alpha_t_cumprod = diffusion_model.alphas[t], diffusion_model.alphas_cumprod[t]
        guided_noise = predicted_noise - torch.sqrt(1. - alpha_t_cumprod) * grad * guidance_scale
        x_t_minus_1_mean = (x_t - (1-alpha_t)/torch.sqrt(1-alpha_t_cumprod) * guided_noise) / torch.sqrt(alpha_t)
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = x_t_minus_1_mean + torch.sqrt(diffusion_model.betas[t]) * noise
        else:
            x_t = x_t_minus_1_mean
    final_indices = torch.argmax(x_t, dim=1)
    return F.one_hot(final_indices, num_classes=NUCLEOTIDES).float().permute(0,2,1)

SAMPLING_BATCH_SIZE = 16
GUIDANCE_STRENGTH = 0.1
rl_diffusion_seqs = guided_sampling(diffusion, real_oracle, batch_size=SAMPLING_BATCH_SIZE, guidance_scale=GUIDANCE_STRENGTH)
baseline_diffusion_seqs = guided_sampling(diffusion, real_oracle, batch_size=SAMPLING_BATCH_SIZE, guidance_scale=0.0)

# ==============================================================================
# 第6部分: 基础评估与可视化
# ==============================================================================
print("\n--- Part 6: Final Evaluation and Visualization ---")
plot_logo(baseline_diffusion_seqs, title='Baseline Diffusion-Generated Sequences (V2)')
plot_logo(rl_diffusion_seqs, title='RL-Guided Diffusion-Generated Sequences (V2)')

print("--- Generating Reward Distribution Plot ---")
with torch.no_grad():
    # 注意: 为了得到更平滑的分布图，可以多次运行生成，或在下方直接加载更多数据
    base_rewards = real_oracle(baseline_diffusion_seqs.to(device)).cpu().numpy().flatten()
    rl_rewards = real_oracle(rl_diffusion_seqs.to(device)).cpu().numpy().flatten()

df = pd.DataFrame({
    'Model': ['Baseline'] * len(base_rewards) + ['RL-Guided'] * len(rl_rewards),
    'Predicted r1 Score': np.concatenate([base_rewards, rl_rewards])
})
plt.figure(figsize=(8, 6))
sns.violinplot(x='Model', y='Predicted r1 Score', data=df, palette="viridis")
plt.title("Distribution of Predicted r1 Scores (Diffusion)")
plt.grid(True, axis='y')
plt.savefig(os.path.join(output_dir, "reward_dist_diffusion.png"), dpi=300)
plt.show()

# ==============================================================================
# 第7部分: 综合评估与可视化 (统一对比)
# ==============================================================================
print("\n--- Part 7: Comprehensive Sequence Evaluation ---")

# 设置评估样本数量
N_EVAL_SAMPLES = 200

# 1. 生成各类型序列用于评估
print("--- Generating sequences for evaluation ---")

# 真实数据
real_seqs_eval = real_seq_strings[:N_EVAL_SAMPLES]
real_one_hot_eval = torch.from_numpy(one_hot_sequences[:N_EVAL_SAMPLES]).to(device)


# 随机序列基线
def generate_random_sequences(num_seqs, seq_len):
    sequences_str = []
    for _ in range(num_seqs):
        seq = "".join(np.random.choice(['A', 'C', 'G', 'T'], size=seq_len))
        sequences_str.append(seq)
    # 将随机序列转换为one-hot编码
    one_hot_seqs = []
    for seq in sequences_str:
        encoded = one_hot_encode(seq)
        if encoded is not None:
            one_hot_seqs.append(encoded)
    if one_hot_seqs:
        one_hot_array = np.array(one_hot_seqs)
        return sequences_str, torch.from_numpy(one_hot_array).to(device)
    else:
        return sequences_str, torch.tensor([]).to(device)


random_seqs_str, random_seqs_one_hot = generate_random_sequences(N_EVAL_SAMPLES, SEQUENCE_LENGTH)


SAMPLING_BATCH_SIZE = 16
GUIDANCE_STRENGTH = 0.1
rl_diffusion_seqs = guided_sampling(diffusion, real_oracle, batch_size=SAMPLING_BATCH_SIZE, guidance_scale=GUIDANCE_STRENGTH)
baseline_diffusion_seqs = guided_sampling(diffusion, real_oracle, batch_size=SAMPLING_BATCH_SIZE, guidance_scale=0.0)


# 辅助函数：将one-hot转回字符串
def one_hot_to_strings(one_hot_tensor):
    if isinstance(one_hot_tensor, torch.Tensor):
        one_hot_tensor = one_hot_tensor.detach().cpu().numpy()

    # 确保形状正确 [batch, channels, length] -> 转换为 [batch, length, channels]
    if one_hot_tensor.shape[1] == 4:  # 如果是 [batch, 4, length]
        one_hot_tensor = np.transpose(one_hot_tensor, (0, 2, 1))

    indices = np.argmax(one_hot_tensor, axis=2)
    return ["".join([idx_to_char.get(i, 'N') for i in seq]) for seq in indices]


# 转换为字符串序列
baseline_seqs_str = one_hot_to_strings(baseline_diffusion_seqs)
rl_seqs_str = one_hot_to_strings(rl_diffusion_seqs)

# --- 1. 奖励分布小提琴图 ---
print("--- Generating Reward Distribution Plot (Violin) ---")
with torch.no_grad():
    # 使用oracle模型预测各序列的r1分数
    rand_preds = real_oracle(
        random_seqs_one_hot).cpu().numpy().flatten() if random_seqs_one_hot.numel() > 0 else np.array([])
    base_preds = real_oracle(baseline_diffusion_seqs.to(device)).cpu().numpy().flatten()
    rl_preds = real_oracle(rl_diffusion_seqs.to(device)).cpu().numpy().flatten()
    real_preds = real_oracle(real_one_hot_eval.to(device)).cpu().numpy().flatten()

# 创建数据框用于绘图
df_violin = pd.DataFrame({
    'Group': ['Real'] * len(real_preds) + ['Random'] * len(rand_preds) + ['Baseline'] * len(base_preds) + [
        'RL-Guided'] * len(rl_preds),
    'Predicted r1 Score': np.concatenate([real_preds, rand_preds, base_preds, rl_preds])
})

plt.figure(figsize=(10, 6))
sns.violinplot(x='Group', y='Predicted r1 Score', data=df_violin, palette="muted")
plt.title("Distribution of Predicted r1 Scores (Diffusion Model)")
plt.savefig(os.path.join(output_dir, "reward_distribution_diffusion.png"), dpi=300)
plt.show()

# --- 2. Levenshtein距离评估 ---
print("\n[Metric 1] Levenshtein Distance")


def calculate_avg_min_levenshtein(generated_seqs, real_seqs):
    total_min_dist = 0
    if not generated_seqs or not real_seqs:
        return float('inf')

    for gen_seq in tqdm(generated_seqs, desc="Calculating Levenshtein"):
        min_dist = min([Levenshtein.distance(gen_seq, real_seq) for real_seq in real_seqs])
        total_min_dist += min_dist
    return total_min_dist / len(generated_seqs)


# 计算各生成序列与真实序列的平均最小Levenshtein距离
dist_random = calculate_avg_min_levenshtein(random_seqs_str, real_seqs_eval)
dist_baseline = calculate_avg_min_levenshtein(baseline_seqs_str, real_seqs_eval)
dist_rl = calculate_avg_min_levenshtein(rl_seqs_str, real_seqs_eval)

print(f"  Avg Min Levenshtein (Random vs Real):   {dist_random:.4f}")
print(f"  Avg Min Levenshtein (Baseline vs Real): {dist_baseline:.4f}")
print(f"  Avg Min Levenshtein (RL-Guided vs Real): {dist_rl:.4f}")

# --- 3. k-mer分布相似性 (Jensen-Shannon Divergence) ---
print("\n[Metric 2] 4-mer JSD")


def get_kmer_distribution(sequences, k=4):
    """计算序列的k-mer分布"""
    all_kmers = []
    for seq in sequences:
        # 提取所有k-mers
        kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        all_kmers.extend(kmers)

    counts = Counter(all_kmers)
    total = sum(counts.values())
    return {kmer: count / total for kmer, count in counts.items()}


# 计算各序列集的k-mer分布
real_kmer_dist = get_kmer_distribution(real_seqs_eval)
random_kmer_dist = get_kmer_distribution(random_seqs_str)
baseline_kmer_dist = get_kmer_distribution(baseline_seqs_str)
rl_kmer_dist = get_kmer_distribution(rl_seqs_str)

# 合并所有k-mer作为特征空间
all_kmers = set(real_kmer_dist.keys()) | set(random_kmer_dist.keys()) | set(baseline_kmer_dist.keys()) | set(
    rl_kmer_dist.keys())
all_kmers = sorted(list(all_kmers))

# 创建概率分布向量
p_real = np.array([real_kmer_dist.get(k, 0) for k in all_kmers])
p_random = np.array([random_kmer_dist.get(k, 0) for k in all_kmers])
p_baseline = np.array([baseline_kmer_dist.get(k, 0) for k in all_kmers])
p_rl = np.array([rl_kmer_dist.get(k, 0) for k in all_kmers])


# 计算Jensen-Shannon散度
def jsd(p, q):
    if len(p) == 0 or len(q) == 0:
        return float('inf')
    return jensenshannon(p, q, base=2.0)


jsd_random = jsd(p_real, p_random)
jsd_baseline = jsd(p_real, p_baseline)
jsd_rl = jsd(p_real, p_rl)

print(f"  4-mer JSD (Random vs Real):   {jsd_random:.4f}")
print(f"  4-mer JSD (Baseline vs Real): {jsd_baseline:.4f}")
print(f"  4-mer JSD (RL-Guided vs Real): {jsd_rl:.4f}")

# --- 4. GC含量分布 ---
print("\n[Metric 3] GC Content Distribution")


def get_gc_content(sequences):
    """计算序列的GC含量"""
    gc_contents = []
    for seq in sequences:
        gc_count = seq.count('G') + seq.count('C')
        gc_content = (gc_count / len(seq)) * 100 if len(seq) > 0 else 0
        gc_contents.append(gc_content)
    return gc_contents


gc_real = get_gc_content(real_seqs_eval)
gc_random = get_gc_content(random_seqs_str)
gc_baseline = get_gc_content(baseline_seqs_str)
gc_rl = get_gc_content(rl_seqs_str)

# 绘制GC含量分布
plt.figure(figsize=(10, 6))
sns.kdeplot(gc_real, label='Real', fill=True)
sns.kdeplot(gc_random, label='Random', fill=True)
sns.kdeplot(gc_baseline, label='Baseline', fill=True)
sns.kdeplot(gc_rl, label='RL-Guided', fill=True)
plt.xlabel('GC Content (%)')
plt.ylabel('Density')
plt.title('GC Content Distribution Comparison')
plt.legend()
plt.savefig(os.path.join(output_dir, "gc_content_distribution_diffusion.png"), dpi=300)
plt.show()

# --- 5. t-SNE可视化 ---
print("\n[Metric 4] t-SNE Visualization")


def get_kmer_features(sequences, k=3):
    """从序列中提取k-mer特征"""
    # 构建k-mer词汇表
    all_kmers = set()
    for seq in sequences:
        kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        all_kmers.update(kmers)

    kmer_list = sorted(list(all_kmers))
    kmer_to_idx = {kmer: i for i, kmer in enumerate(kmer_list)}

    # 构建特征矩阵
    features = np.zeros((len(sequences), len(kmer_to_idx)))
    for i, seq in enumerate(sequences):
        kmers = [seq[j:j + k] for j in range(len(seq) - k + 1)]
        counts = Counter(kmers)
        total = sum(counts.values())
        for kmer, count in counts.items():
            if kmer in kmer_to_idx:
                features[i, kmer_to_idx[kmer]] = count / total if total > 0 else 0
    return features


# 准备t-SNE数据
all_seqs = real_seqs_eval + random_seqs_str + baseline_seqs_str + rl_seqs_str
labels = ['Real'] * len(real_seqs_eval) + ['Random'] * len(random_seqs_str) + ['Baseline'] * len(baseline_seqs_str) + [
    'RL-Guided'] * len(rl_seqs_str)

# 提取特征
features = get_kmer_features(all_seqs, k=3)

# 执行t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
embeddings = tsne.fit_transform(features)

# 绘制t-SNE图
plt.figure(figsize=(10, 8))
for label in ['Real', 'Random', 'Baseline', 'RL-Guided']:
    indices = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(embeddings[indices, 0], embeddings[indices, 1], label=label, alpha=0.7)
plt.legend()
plt.title('t-SNE Visualization of Sequence Distributions')
plt.savefig(os.path.join(output_dir, "tsne_diffusion.png"), dpi=300)
plt.show()

# --- 6. FID (Fréchet Inception Distance) ---
print("\n[Metric 5] FID Calculation")


@torch.no_grad()
def get_embeddings(one_hot_seqs, oracle_model, batch_size=32):
    """使用oracle模型获取序列嵌入"""
    embeddings = []
    for i in range(0, one_hot_seqs.shape[0], batch_size):
        batch = one_hot_seqs[i:i + batch_size]
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch).to(device)
        emb = oracle_model.get_embedding(batch)
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def calculate_fid(embeddings1, embeddings2):
    """计算FID分数"""
    if len(embeddings1) == 0 or len(embeddings2) == 0:
        return float('inf')

    mu1, sigma1 = np.mean(embeddings1, axis=0), np.cov(embeddings1, rowvar=False)
    mu2, sigma2 = np.mean(embeddings2, axis=0), np.cov(embeddings2, rowvar=False)

    # 处理奇异矩阵
    if np.isnan(sigma1).any() or np.isinf(sigma1).any() or np.isnan(sigma2).any() or np.isinf(sigma2).any():
        return float('inf')

    try:
        # 计算FID
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid
    except:
        return float('inf')


# 计算嵌入
real_embs = get_embeddings(real_one_hot_eval, real_oracle)
baseline_embs = get_embeddings(baseline_diffusion_seqs, real_oracle)
rl_embs = get_embeddings(rl_diffusion_seqs, real_oracle)

# 计算FID
fid_baseline = calculate_fid(real_embs, baseline_embs)
fid_rl = calculate_fid(real_embs, rl_embs)

print(f"  FID (Baseline vs Real): {fid_baseline:.4f}")
print(f"  FID (RL-Guided vs Real): {fid_rl:.4f}")

# --- 最终评分汇总 ---
print("\n=== Final Evaluation Results ===")
print("Metric Summary (Lower is better for distance metrics):")
print(f"Levenshtein Distance - Random: {dist_random:.4f}, Baseline: {dist_baseline:.4f}, RL-Guided: {dist_rl:.4f}")
print(f"4-mer JSD - Random: {jsd_random:.4f}, Baseline: {jsd_baseline:.4f}, RL-Guided: {jsd_rl:.4f}")
print(f"FID - Baseline: {fid_baseline:.4f}, RL-Guided: {fid_rl:.4f}")

print("\nReward Scores (Higher is better):")
print(f"Random: {np.mean(rand_preds):.4f}")
print(f"Baseline: {np.mean(base_preds):.4f}")
print(f"RL-Guided: {np.mean(rl_preds):.4f}")
print(f"Real: {np.mean(real_preds):.4f}")