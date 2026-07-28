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
output_dir = "output_optimized_cosine_noisy_oracle_plt"  # 修改输出目录名以区分
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

# ==============================================================================
# 第7部分: 遗传算法 (Genetic Algorithm) 基线实现
# ==============================================================================
print("\n--- Part 7: Setting up Genetic Algorithm Baseline ---")


class GeneticAlgorithm:
    def __init__(self, oracle_model, seq_len=17, pop_size=100, mutation_rate=0.1):
        self.oracle = oracle_model
        self.seq_len = seq_len
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.nucleotides = ['A', 'C', 'G', 'T']
        self.char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def encode(self, seqs):
        # 将字符串列表转换为 One-hot Tensor (N, 4, L)
        batch = np.zeros((len(seqs), 4, self.seq_len), dtype=np.float32)
        for i, seq in enumerate(seqs):
            for j, char in enumerate(seq):
                if char in self.char_map:
                    batch[i, self.char_map[char], j] = 1.0
        return torch.tensor(batch).to(device)

    def initial_population(self):
        return ["".join(random.choices(self.nucleotides, k=self.seq_len)) for _ in range(self.pop_size)]

    def fitness(self, population):
        # 使用 Oracle 计算适应度
        with torch.no_grad():
            one_hots = self.encode(population)
            # 确保 Oracle 处于 eval 模式
            self.oracle.eval()
            scores = self.oracle(one_hots).cpu().numpy().flatten()
        return scores

    def mutate(self, seq):
        seq_list = list(seq)
        if random.random() < self.mutation_rate:
            pos = random.randint(0, self.seq_len - 1)
            seq_list[pos] = random.choice(self.nucleotides)
        return "".join(seq_list)

    def crossover(self, parent1, parent2):
        point = random.randint(1, self.seq_len - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

    def run(self, generations=50):
        population = self.initial_population()
        best_score = -999

        # 进化循环
        for _ in tqdm(range(generations), desc="Running GA", leave=False):
            scores = self.fitness(population)
            current_best = np.max(scores)
            if current_best > best_score:
                best_score = current_best

            # 选择策略: 保留前 50%
            sorted_indices = np.argsort(scores)[::-1]
            survivors = [population[i] for i in sorted_indices[:self.pop_size // 2]]

            # 繁殖下一代
            next_gen = survivors[:]
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(survivors, 2)
                c1, c2 = self.crossover(p1, p2)
                next_gen.append(self.mutate(c1))
                if len(next_gen) < self.pop_size:
                    next_gen.append(self.mutate(c2))

            population = next_gen

        return population


# ==============================================================================
# 第8部分: 生成对比数据与计算指标
# ==============================================================================
print("\n--- Part 8: Generating Benchmarks & Calculating Metrics ---")

# 参数设置
N_EVAL = 200  # 每组评估样本数
BATCH_SIZE_EVAL = 64

# 1. Real Data (从训练集中随机采样)
real_indices = np.random.choice(len(one_hot_sequences), N_EVAL, replace=False)
real_seqs_tensor = torch.from_numpy(one_hot_sequences[real_indices]).to(device)
real_strs = to_str(real_seqs_tensor)
with torch.no_grad():
    real_scores = real_oracle(real_seqs_tensor).cpu().numpy().flatten()

# 2. Random Data (随机生成)
random_strs = ["".join(random.choices(['A', 'C', 'G', 'T'], k=SEQUENCE_LENGTH)) for _ in range(N_EVAL)]
random_tensor = GeneticAlgorithm(real_oracle).encode(random_strs)
with torch.no_grad():
    random_scores = real_oracle(random_tensor).cpu().numpy().flatten()

# 3. Genetic Algorithm (运行 GA)
print("Generating GA sequences...")
ga_solver = GeneticAlgorithm(real_oracle, seq_len=SEQUENCE_LENGTH, pop_size=N_EVAL, mutation_rate=0.1)
ga_strs = ga_solver.run(generations=50)  # 运行 50 代
ga_tensor = ga_solver.encode(ga_strs)
with torch.no_grad():
    ga_scores = real_oracle(ga_tensor).cpu().numpy().flatten()

# 4. RNADGG (我们的模型)
print("Generating RNADGG sequences...")
rnadgg_tensor_list = []
# 分批生成以防显存溢出
num_batches = math.ceil(N_EVAL / BATCH_SIZE_EVAL)
for _ in range(num_batches):
    batch = guided_sampling(diffusion, real_oracle, batch_size=BATCH_SIZE_EVAL, guidance_scale=2.0)  # 强引导
    rnadgg_tensor_list.append(batch)
rnadgg_tensor = torch.cat(rnadgg_tensor_list, dim=0)[:N_EVAL]
rnadgg_strs = to_str(rnadgg_tensor)
with torch.no_grad():
    rnadgg_scores = real_oracle(rnadgg_tensor).cpu().numpy().flatten()


# --- 计算多样性 (Diversity) ---
def calculate_diversity(seqs_list):
    # 计算集合内的平均成对 Levenshtein 距离
    # 为了速度，随机采样最多 1000 对进行计算
    if len(seqs_list) < 2: return 0
    total_dist = 0
    count = 0
    limit = 2000
    for _ in range(limit):
        s1, s2 = random.sample(seqs_list, 2)
        total_dist += Levenshtein.distance(s1, s2)
        count += 1
    return total_dist / count


print("Calculating Diversity...")
div_real = calculate_diversity(real_strs)
div_rand = calculate_diversity(random_strs)
div_ga = calculate_diversity(ga_strs)
div_rnadgg = calculate_diversity(rnadgg_strs)

print(f"\n--- Benchmark Results ---")
print(f"Real Data:   Mean Score={np.mean(real_scores):.4f}, Diversity={div_real:.2f}")
print(f"Random:      Mean Score={np.mean(random_scores):.4f}, Diversity={div_rand:.2f}")
print(f"Genetic Alg: Mean Score={np.mean(ga_scores):.4f}, Diversity={div_ga:.2f}")
print(f"RNADGG:      Mean Score={np.mean(rnadgg_scores):.4f}, Diversity={div_rnadgg:.2f}")

# ==============================================================================
# 第9部分: 可视化 (Violin Plot & Scatter Plot)
# ==============================================================================
print("\n--- Part 9: Visualizing Results ---")

# 1. 分数分布对比 (Violin Plot)
data_viz = pd.DataFrame({
    'Score': np.concatenate([real_scores, random_scores, ga_scores, rnadgg_scores]),
    'Method': (['Real Data'] * len(real_scores) +
               ['Random'] * len(random_scores) +
               ['Genetic Algo'] * len(ga_scores) +
               ['RNADGG (Ours)'] * len(rnadgg_scores))
})

plt.figure(figsize=(10, 6))
# 使用 muted 调色板
sns.violinplot(x='Method', y='Score', data=data_viz, palette="muted", inner='quartile', cut=0)
plt.title("Functional Score Distribution Comparison")
plt.ylabel("Predicted Activity (Oracle Score)")
plt.savefig(os.path.join(output_dir, "comparison_violin.png"), dpi=300)
plt.show()

# 2. 多样性 vs 分数 (Scatter Plot - Pareto Frontier)
methods = ['Real', 'Random', 'Genetic Algo', 'RNADGG']
mean_scores = [np.mean(real_scores), np.mean(random_scores), np.mean(ga_scores), np.mean(rnadgg_scores)]
diversities = [div_real, div_rand, div_ga, div_rnadgg]
colors = ['gray', 'blue', 'orange', 'red']  # RNADGG 用红色突出

plt.figure(figsize=(8, 6))
for i, method in enumerate(methods):
    plt.scatter(diversities[i], mean_scores[i], s=250, label=method, color=colors[i], alpha=0.8, edgecolors='black')
    # 添加文字标签
    plt.text(diversities[i], mean_scores[i] + 0.02, method, ha='center', fontsize=9, fontweight='bold')

plt.title("Diversity vs. Functionality Trade-off")
plt.xlabel("Sequence Diversity (Avg Levenshtein Dist)")
plt.ylabel("Average Functional Score")
plt.grid(True, linestyle='--', alpha=0.5)

# 绘制理想区域箭头
max_div = max(diversities)
max_score = max(mean_scores)
plt.annotate('Ideal Region', xy=(max_div, max_score), xytext=(max_div - 2, max_score - 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

plt.savefig(os.path.join(output_dir, "diversity_vs_score.png"), dpi=300)
plt.show()

# 3. 绘制 Sequence Logos
# 对比 GA 和 RNADGG 的 Top 序列模式
plot_logo(ga_tensor[:50], title="Genetic Algorithm (Top 50)")
plot_logo(rnadgg_tensor[:50], title="RNADGG Generated (Top 50)")

print(f"All visualizations saved to {output_dir}")