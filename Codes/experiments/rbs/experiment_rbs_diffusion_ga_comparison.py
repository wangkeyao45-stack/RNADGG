# -*- coding: utf-8 -*-
# 文件名: main_model.py
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


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sns.set_theme(style="whitegrid")
output_dir = "output_plots_final_main"
os.makedirs(output_dir, exist_ok=True)


def plot_logo(one_hot_seqs, title=''):
    """使用logomaker库绘制专业的序列标识图"""
    if isinstance(one_hot_seqs, torch.Tensor):
        one_hot_seqs = one_hot_seqs.detach().cpu().numpy()
    if one_hot_seqs.shape[-1] != 4:
        one_hot_seqs = np.transpose(one_hot_seqs, (0, 2, 1))

    pwm = np.mean(one_hot_seqs, axis=0)
    pwm_df = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])
    seq_length = pwm_df.shape[0]
    fig_width = max(10, seq_length * 0.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 2.5))

    logo = logomaker.Logo(pwm_df, ax=ax, shade_below=0.5, fade_below=0.5, font_name='sans-serif',
                          color_scheme='colorblind_safe')
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.ax.set_ylabel("Probability", labelpad=2, fontsize=10)
    logo.ax.set_ylim([0, 1])
    logo.ax.set_title(title, fontsize=12, pad=10)

    sanitized_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"logo_{sanitized_title}.png"), dpi=300, bbox_inches='tight')
    if __name__ == "__main__":  # 只有在主程序运行时才显示，被调用时不显示
        plt.show()
    plt.close()


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
        return F.mse_loss(predicted_noise, noise)


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
# 第2部分: 数据加载与Oracle训练 (全局执行，以便其他脚本调用)
# ==============================================================================
print("--- [main_model.py] 初始化数据与Oracle ---")

if not os.path.exists(CSV_FILE):
    # 模拟数据生成
    print(f"提示: '{CSV_FILE}' 未找到，使用模拟数据。")
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


NUM_SAMPLES_TO_USE = 260000
df_subset = df.head(NUM_SAMPLES_TO_USE)
sequences = df_subset['序列'].tolist()
rl_scores = df_subset['rl'].values.astype(np.float32)
processed_data = [(one_hot_encode(s), r, s) for s, r in zip(sequences, rl_scores) if one_hot_encode(s) is not None]
one_hot_sequences = np.array([item[0] for item in processed_data])
scores = np.array([item[1] for item in processed_data])
real_seq_strings = [item[2] for item in processed_data]
print(f"数据加载完成。Shape: {one_hot_sequences.shape}")

# 训练 Real Oracle (如果被import调用，也会执行，这保证了导入的oracle是可用的)
print("正在初始化并训练 Oracle (作为评估基准)...")
X_train, X_val, y_train, y_val = train_test_split(one_hot_sequences, scores, test_size=0.2, random_state=42)
oracle_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32,
                                 shuffle=True)
real_oracle = OracleCNN().to(device)
oracle_optimizer = optim.Adam(real_oracle.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 快速训练 Oracle (为了演示设为20，实际上建议50)
ORACLE_EPOCHS = 20
for epoch in range(ORACLE_EPOCHS):
    real_oracle.train()
    for seqs, labels in oracle_train_loader:
        oracle_optimizer.zero_grad()
        loss = criterion(real_oracle(seqs.to(device)).squeeze(), labels.to(device))
        loss.backward()
        oracle_optimizer.step()
real_oracle.eval()
print("Oracle 准备就绪。")


# ==============================================================================
# 第3部分: 辅助类 (Genetic Algorithm) & 生成函数
# ==============================================================================
class GeneticAlgorithm:
    def __init__(self, oracle_model, seq_len=17, pop_size=200, mutation_rate=0.1):
        self.oracle = oracle_model
        self.seq_len = seq_len
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.n_nucleotides = 4

    def initial_population(self):
        indices = torch.randint(0, self.n_nucleotides, (self.pop_size, self.seq_len), device=device)
        return F.one_hot(indices, num_classes=self.n_nucleotides).float().permute(0, 2, 1)

    def optimize(self, generations=50):
        population = self.initial_population()
        for g in tqdm(range(generations), desc="GA Optimization", leave=False):
            with torch.no_grad():
                scores = self.oracle(population).flatten()
            sorted_idx = torch.argsort(scores, descending=True)
            num_elites = int(self.pop_size * 0.2)
            elites = population[sorted_idx[:num_elites]]
            parents = population[sorted_idx[:self.pop_size // 2]]

            idx_1 = torch.randint(0, len(parents), (self.pop_size - num_elites,), device=device)
            idx_2 = torch.randint(0, len(parents), (self.pop_size - num_elites,), device=device)
            p1, p2 = parents[idx_1], parents[idx_2]

            crossover_pt = torch.randint(1, self.seq_len, (self.pop_size - num_elites, 1, 1), device=device)
            mask = torch.arange(self.seq_len, device=device).view(1, 1, -1) < crossover_pt
            offspring = torch.where(mask, p1, p2)

            mutation_mask = torch.rand(offspring.shape[0], self.seq_len, device=device) < self.mutation_rate
            if mutation_mask.any():
                random_indices = torch.randint(0, self.n_nucleotides, (offspring.shape[0], self.seq_len), device=device)
                random_onehot = F.one_hot(random_indices, num_classes=self.n_nucleotides).float().permute(0, 2, 1)
                mask_expanded = mutation_mask.unsqueeze(1).expand_as(offspring)
                offspring = torch.where(mask_expanded, random_onehot, offspring)
            population = torch.cat([elites, offspring], dim=0)
        return population


def guided_sampling(diffusion_model, oracle_model, batch_size, guidance_scale=0.01, track_gradients=False):
    x_t = torch.randn((batch_size, NUCLEOTIDES, SEQUENCE_LENGTH), device=device)
    gradient_logs = []
    for t in tqdm(reversed(range(diffusion_model.T)), desc="Sampling", total=diffusion_model.T, leave=False):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        grad = torch.zeros_like(x_t)
        if guidance_scale != 0:
            with torch.enable_grad():
                x_t.requires_grad_()
                scores = oracle_model(x_t).sum()
                g = torch.autograd.grad(scores, x_t)[0]
                if track_gradients: gradient_logs.append(g.norm().item() / batch_size)
                grad = g.clamp(-1, 1)
        elif track_gradients:
            gradient_logs.append(0)

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


def one_hot_to_strings(one_hot_tensor):
    if isinstance(one_hot_tensor, torch.Tensor): one_hot_tensor = one_hot_tensor.detach().cpu().numpy()
    if one_hot_tensor.shape[1] == 4: one_hot_tensor = np.transpose(one_hot_tensor, (0, 2, 1))
    indices = np.argmax(one_hot_tensor, axis=2)
    return ["".join([idx_to_char.get(i, 'N') for i in seq]) for seq in indices]


def save_sequences_to_txt(filename, sequences, scores=None):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        f.write("Sequence,Predicted_Score\n")
        for i, seq in enumerate(sequences):
            score_str = f"{scores[i]:.4f}" if scores is not None else "N/A"
            f.write(f"{seq},{score_str}\n")
    print(f"Saved to {filepath}")


# ==============================================================================
# 第4部分: 主执行逻辑 (被 import 时不会执行)
# ==============================================================================
if __name__ == "__main__":
    print("\n--- [Part 4] 开始训练 Diffusion 模型 ---")
    diffusion_train_loader = DataLoader(TensorDataset(torch.from_numpy(one_hot_sequences)), batch_size=64, shuffle=True)
    unet = UNet().to(device)
    diffusion = Diffusion(unet)
    diffusion_optimizer = optim.Adam(unet.parameters(), lr=1e-4)

    DIFFUSION_EPOCHS = 50  # 演示用，建议 100
    for epoch in range(DIFFUSION_EPOCHS):
        pbar = tqdm(diffusion_train_loader, desc=f"Diff Epoch {epoch + 1}/{DIFFUSION_EPOCHS}", leave=False)
        for i, (real_seqs,) in enumerate(pbar):
            diffusion_optimizer.zero_grad()
            loss = diffusion.train_step(real_seqs.to(device))
            loss.backward()
            diffusion_optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    print("Diffusion 训练完成。")

    # ==============================================================================
    # 第5部分: 综合评估与对比
    # ==============================================================================
    print("\n--- [Part 5] 综合评估 (对比 GA / Diffusion / RL) ---")
    N_EVAL = 200
    BATCH_SIZE = 32
    GUIDANCE = 0.5

    # 1. 准备数据
    print("生成评估数据...")
    # Real
    real_idx = np.random.choice(len(one_hot_sequences), N_EVAL, replace=False)
    real_oh_eval = torch.from_numpy(one_hot_sequences[real_idx]).to(device)
    real_str_eval = [real_seq_strings[i] for i in real_idx]

    # Random
    rand_seqs = ["".join(np.random.choice(['A', 'C', 'G', 'T'], SEQUENCE_LENGTH)) for _ in range(N_EVAL)]
    rand_oh = torch.from_numpy(np.array([one_hot_encode(s) for s in rand_seqs])).float().to(device)

    # Baseline Diffusion
    base_oh = generate_sequences_in_batches(diffusion, real_oracle, N_EVAL, BATCH_SIZE, 0.0)
    base_str = one_hot_to_strings(base_oh)

    # RL-Guided Diffusion
    rl_oh = generate_sequences_in_batches(diffusion, real_oracle, N_EVAL, BATCH_SIZE, GUIDANCE)
    rl_str = one_hot_to_strings(rl_oh)

    # Traditional GA
    print("运行遗传算法 (Traditional GA)...")
    ga_solver = GeneticAlgorithm(real_oracle, seq_len=SEQUENCE_LENGTH, pop_size=N_EVAL)
    ga_oh = ga_solver.optimize(generations=50)
    ga_str = one_hot_to_strings(ga_oh)

    # 2. 计算指标
    print("\n计算对比指标...")


    def get_kmer_dist(seqs, k=4):
        cnt = Counter([s[i:i + k] for s in seqs for i in range(len(s) - k + 1)])
        total = sum(cnt.values())
        return {k: v / total for k, v in cnt.items()}


    real_dist = get_kmer_dist(real_seq_strings)  # 使用全量真实数据作为分布基准
    all_k = sorted(list(real_dist.keys()))


    def calc_metrics(name, gen_str, gen_oh):
        # Levenshtein (Diversity/Novelty)
        lev = sum([min([Levenshtein.distance(g, r) for r in real_str_eval]) for g in gen_str]) / len(gen_str)
        # JSD (Distribution)
        d = get_kmer_dist(gen_str)
        p = np.array([real_dist.get(k, 0) for k in all_k])
        q = np.array([d.get(k, 0) for k in all_k])
        jsd = jensenshannon(p, q, base=2.0)
        # FID (Quality)
        with torch.no_grad():
            real_embs = real_oracle.get_embedding(real_oh_eval).cpu().numpy()
            gen_embs = real_oracle.get_embedding(gen_oh).cpu().numpy()
        mu1, sig1 = np.mean(real_embs, axis=0), np.cov(real_embs, rowvar=False)
        mu2, sig2 = np.mean(gen_embs, axis=0), np.cov(gen_embs, rowvar=False)
        diff = mu1 - mu2
        covmean = sqrtm(sig1.dot(sig2))
        if np.iscomplexobj(covmean): covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sig1 + sig2 - 2 * covmean)
        return lev, jsd, fid


    datasets = [('Random', rand_seqs, rand_oh), ('Baseline', base_str, base_oh),
                ('Genetic Alg', ga_str, ga_oh), ('RL-Guided', rl_str, rl_oh)]

    print(f"{'Method':<15} | {'LevDist':<8} | {'JSD':<8} | {'FID':<8}")
    print("-" * 50)
    for name, s_str, s_oh in datasets:
        l, j, f = calc_metrics(name, s_str, s_oh)
        print(f"{name:<15} | {l:<8.2f} | {j:<8.4f} | {f:<8.4f}")

    # 3. 绘图: Violin Plot
    with torch.no_grad():
        s_real = real_oracle(real_oh_eval).cpu().numpy().flatten()
        s_rand = real_oracle(rand_oh).cpu().numpy().flatten()
        s_base = real_oracle(base_oh).cpu().numpy().flatten()
        s_ga = real_oracle(ga_oh).cpu().numpy().flatten()
        s_rl = real_oracle(rl_oh).cpu().numpy().flatten()

    df_viol = pd.DataFrame({
        'Method': ['Real'] * N_EVAL + ['Random'] * N_EVAL + ['Baseline'] * N_EVAL + ['Genetic Alg'] * N_EVAL + [
            'RL-Guided'] * N_EVAL,
        'Score': np.concatenate([s_real, s_rand, s_base, s_ga, s_rl])
    })
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Method', y='Score', data=df_viol, palette="Set3", hue='Method', legend=False)
    plt.title("Reward Distribution Comparison")
    plt.savefig(os.path.join(output_dir, "comparison_violin.png"), dpi=300)
    plt.show()

    # 4. 绘图: Logos
    print("绘制 Sequence Logos...")
    plot_logo(real_oh_eval, "Real Data (Subset)")
    plot_logo(rand_oh, "Random Baseline")
    plot_logo(base_oh, "Diffusion Baseline")
    plot_logo(ga_oh, "Traditional Genetic Algorithm")
    plot_logo(rl_oh, "RL-Guided Diffusion")

    # 5. 保存结果
    save_sequences_to_txt("final_seqs_ga.txt", ga_str, s_ga)
    save_sequences_to_txt("final_seqs_rl.txt", rl_str, s_rl)

    print("\nMain execution finished successfully.")