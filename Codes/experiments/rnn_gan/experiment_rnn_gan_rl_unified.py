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
# 第0部分: 环境设置、随机种子和辅助函数
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

set_seed(42) # 统一设置种子

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sns.set_theme(style="whitegrid")
output_dir = "output_plots_rnn_rl_finetune"
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
# 第1部分: PyTorch 模型定义 (RNN)
# ==============================================================================
LATENT_DIM = 128
SEQUENCE_LENGTH = 50
NUCLEOTIDES = 4
VOCAB_SIZE = NUCLEOTIDES + 1
SOS_TOKEN_IDX = NUCLEOTIDES
LSTM_HIDDEN_DIM = 256
CSV_FILE = "processed_data.csv"

class GeneratorRNN(nn.Module):
    def __init__(self):
        super(GeneratorRNN, self).__init__()
        self.latent_to_hidden = nn.Linear(LATENT_DIM, LSTM_HIDDEN_DIM)
        self.latent_to_cell = nn.Linear(LATENT_DIM, LSTM_HIDDEN_DIM)
        self.embedding = nn.Embedding(VOCAB_SIZE, LSTM_HIDDEN_DIM)
        self.lstm = nn.LSTM(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, batch_first=True)
        self.hidden_to_logits = nn.Linear(LSTM_HIDDEN_DIM, NUCLEOTIDES)

    def forward(self, z, sequence_length=SEQUENCE_LENGTH):
        batch_size = z.size(0)
        h_0 = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)
        c_0 = torch.tanh(self.latent_to_cell(z)).unsqueeze(0)
        hidden = (h_0, c_0)
        sos_token = torch.full((batch_size, 1), SOS_TOKEN_IDX, dtype=torch.long, device=z.device)
        input_token_emb = self.embedding(sos_token)
        all_logits = []
        all_indices = []
        for i in range(sequence_length):
            output, hidden = self.lstm(input_token_emb, hidden)
            logits = self.hidden_to_logits(output.squeeze(1))
            all_logits.append(logits)
            probs = F.softmax(logits, dim=-1)
            indices = torch.multinomial(probs, 1)
            all_indices.append(indices)
            input_token = indices
            input_token_emb = self.embedding(input_token)
        all_logits = torch.stack(all_logits, dim=1)
        all_indices = torch.cat(all_indices, dim=1)
        return all_logits, all_indices

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(NUCLEOTIDES, 128, kernel_size=5, padding='same'),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Conv1d(128, 64, kernel_size=5, padding='same'),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(64 * SEQUENCE_LENGTH, 1)
        )
    def forward(self, seq): return self.model(seq)

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
        if char in char_map: encoded[char_map[char], i] = 1.0
        else: return None
    return encoded

df = pd.read_csv(CSV_FILE)
# --- 统一使用20000条数据 ---
NUM_SAMPLES_TO_USE = 280000
df_subset = df.head(NUM_SAMPLES_TO_USE)
print(f"--- Using a subset of {NUM_SAMPLES_TO_USE} samples. ---")

sequences = df_subset['序列'].tolist()
r1_scores = df_subset['r1'].values.astype(np.float32)
processed_data = [(one_hot_encode(s), r, s) for s, r in zip(sequences, r1_scores) if one_hot_encode(s) is not None]
one_hot_sequences = np.array([item[0] for item in processed_data])
r1_scores = np.array([item[1] for item in processed_data])
real_seq_strings = [item[2] for item in processed_data]
print(f"Data loaded and encoded. Final Shape: {one_hot_sequences.shape}")

# ==============================================================================
# 第3部分: 训练预言机
# ==============================================================================
print("\n--- Part 3: Training the Real Oracle Model ---")
X_train, X_val, y_train, y_val = train_test_split(one_hot_sequences, r1_scores, test_size=0.2, random_state=42)
oracle_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32, shuffle=True)
oracle_val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=32)
real_oracle = OracleCNN().to(device)
oracle_optimizer = optim.Adam(real_oracle.parameters(), lr=1e-3)
criterion = nn.MSELoss()
ORACLE_EPOCHS = 20
for epoch in range(ORACLE_EPOCHS):
    real_oracle.train()
    train_loss = 0
    for seqs, labels in oracle_train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        oracle_optimizer.zero_grad()
        outputs = real_oracle(seqs).squeeze()
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        oracle_optimizer.step()
        train_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Oracle Epoch [{epoch+1}/{ORACLE_EPOCHS}], Train Loss: {train_loss/len(oracle_train_loader):.4f}")
print("--- Real Oracle training finished. ---")
real_oracle.eval()
r1_mean, r1_std = torch.tensor(r1_scores.mean(), device=device), torch.tensor(r1_scores.std(), device=device)
def get_reward_oracle(sequences_n_c_l):
    with torch.no_grad():
        pred_r1 = real_oracle(sequences_n_c_l).squeeze()
        if pred_r1.ndim == 0: pred_r1 = pred_r1.unsqueeze(0)
        norm_r1 = (pred_r1 - r1_mean) / (r1_std + 1e-9)
        return norm_r1

# ==============================================================================
# 第4部分: GAN 预训练
# ==============================================================================
print("\n--- Part 4: Training Baseline GAN Model ---")
gan_train_loader = DataLoader(TensorDataset(torch.from_numpy(one_hot_sequences)), batch_size=64, shuffle=True)
base_generator = GeneratorRNN().to(device)
discriminator = Discriminator().to(device)
g_optimizer = optim.Adam(base_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
GAN_EPOCHS = 60
for epoch in range(GAN_EPOCHS):
    pbar = tqdm(gan_train_loader, desc=f"GAN Epoch {epoch+1}/{GAN_EPOCHS}")
    for i, (real_seqs,) in enumerate(pbar):
        batch_size = real_seqs.size(0)
        real_seqs = real_seqs.to(device)
        d_optimizer.zero_grad()
        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        _, generated_indices = base_generator(noise)
        generated_one_hot = F.one_hot(generated_indices, num_classes=NUCLEOTIDES).float().permute(0, 2, 1)
        real_output = discriminator(real_seqs)
        fake_output = discriminator(generated_one_hot.detach())
        d_loss = torch.mean(torch.relu(1.0 - real_output)) + torch.mean(torch.relu(1.0 + fake_output))
        d_loss.backward()
        d_optimizer.step()
        g_optimizer.zero_grad()
        fake_output = discriminator(generated_one_hot)
        g_loss = -torch.mean(fake_output)
        g_loss.backward()
        g_optimizer.step()
        pbar.set_postfix(D_Loss=f"{d_loss.item():.4f}", G_Loss=f"{g_loss.item():.4f}")
print("--- GAN Training Finished. ---")

# ==============================================================================
# 第5部分: RL模型微调 (Direct Weight Update)
# ==============================================================================
print("\n--- Part 5: RL Model Fine-Tuning ---")
import copy
rl_tuned_generator = copy.deepcopy(base_generator).to(device)
rl_optimizer = optim.Adam(rl_tuned_generator.parameters(), lr=5e-5)
RL_EPOCHS = 100
RL_BATCH_SIZE = 64
rl_history = []

for epoch in tqdm(range(RL_EPOCHS), desc="RL Fine-Tuning"):
    rl_tuned_generator.train()
    noise = torch.randn(RL_BATCH_SIZE, LATENT_DIM, device=device)
    all_logits, generated_indices = rl_tuned_generator(noise)
    generated_one_hot = F.one_hot(generated_indices, num_classes=NUCLEOTIDES).float().permute(0, 2, 1)
    rewards = get_reward_oracle(generated_one_hot)
    log_probs = F.log_softmax(all_logits, dim=-1)
    action_log_probs = log_probs.gather(2, generated_indices.unsqueeze(-1)).squeeze(-1)
    rl_loss = -torch.mean(torch.sum(action_log_probs, dim=1) * (rewards - rewards.mean()))
    rl_optimizer.zero_grad()
    rl_loss.backward()
    rl_optimizer.step()
    rl_history.append(rewards.mean().item())

print("--- RL Fine-Tuning Finished ---")
rl_tuned_generator.eval()

# ==============================================================================
# 第6部分: 生成随机序列
# ==============================================================================
print("\n--- Part 6: Generating Random Sequences Baseline ---")
def generate_random_sequences(num_seqs, seq_len):
    sequences_str = []
    for _ in range(num_seqs):
        seq = "".join(np.random.choice(['A', 'C', 'G', 'T'], size=seq_len))
        sequences_str.append(seq)
    one_hot = np.array([one_hot_encode(s) for s in sequences_str])
    return sequences_str, torch.from_numpy(one_hot).to(device)

# ==============================================================================
# 第7部分: 综合评估与可视化 (统一对比)
# ==============================================================================
print("\n--- Part 7: Comprehensive Sequence Evaluation ---")
N_EVAL_SAMPLES = 200
base_generator.eval()
with torch.no_grad():
    # 1. 基线GAN
    noise = torch.randn(N_EVAL_SAMPLES, LATENT_DIM, device=device)
    _, baseline_indices = base_generator(noise)
    baseline_seqs_one_hot = F.one_hot(baseline_indices, num_classes=NUCLEOTIDES).float().permute(0,2,1)
    # 2. RL微调后
    noise = torch.randn(N_EVAL_SAMPLES, LATENT_DIM, device=device)
    _, rl_indices = rl_tuned_generator(noise)
    rl_tuned_seqs_one_hot = F.one_hot(rl_indices, num_classes=NUCLEOTIDES).float().permute(0,2,1)
# 3. 随机
random_seqs_str, random_seqs_one_hot = generate_random_sequences(N_EVAL_SAMPLES, SEQUENCE_LENGTH)
# 4. 真实
real_seqs_eval = real_seq_strings[:N_EVAL_SAMPLES]
real_one_hot_eval = torch.from_numpy(one_hot_sequences[:N_EVAL_SAMPLES]).to(device)

def one_hot_to_strings(one_hot_tensor):
    indices = torch.argmax(one_hot_tensor.permute(0, 2, 1), dim=2)
    return ["".join([idx_to_char.get(i.item(), 'N') for i in seq]) for seq in indices]

baseline_seqs_str = one_hot_to_strings(baseline_seqs_one_hot)
rl_tuned_seqs_str = one_hot_to_strings(rl_tuned_seqs_one_hot)

# --- 1. 绘制 Sequence Logo ---
print("--- Generating Sequence Logos ---")
plot_logo(baseline_seqs_one_hot, title='Baseline RNN-GAN Sequences')
plot_logo(rl_tuned_seqs_one_hot, title='RL-Tuned RNN Sequences')

# --- 2. 绘制小提琴图 (Violin Plot) ---
print("--- Generating Reward Distribution Plot (Violin) ---")
with torch.no_grad():
    rand_preds = real_oracle(random_seqs_one_hot).cpu().numpy().flatten()
    base_preds = real_oracle(baseline_seqs_one_hot.to(device)).cpu().numpy().flatten()
    rl_preds = real_oracle(rl_tuned_seqs_one_hot.to(device)).cpu().numpy().flatten()
    real_preds = real_oracle(real_one_hot_eval).cpu().numpy().flatten()

df_violin = pd.DataFrame({
    'Group': ['Random']*len(rand_preds) + ['Baseline']*len(base_preds) + ['RL-Tuned']*len(rl_preds) + ['Real']*len(real_preds),
    'Predicted r1 Score': np.concatenate([rand_preds, base_preds, rl_preds, real_preds])
})
plt.figure(figsize=(10, 6))
sns.violinplot(x='Group', y='Predicted r1 Score', data=df_violin, palette="muted")
plt.title("Distribution of Predicted r1 Scores")
plt.savefig(os.path.join(output_dir, "reward_dist_violin.png"), dpi=300)
plt.show()

print("\n[Metric 1] Levenshtein Distance")
def calculate_avg_min_levenshtein(generated_seqs, real_seqs):
    total_min_dist = 0
    if not generated_seqs or not real_seqs: return float('inf')
    for gen_seq in tqdm(generated_seqs, desc="Levenshtein"):
        min_dist = min([Levenshtein.distance(gen_seq, real_seq) for real_seq in real_seqs])
        total_min_dist += min_dist
    return total_min_dist / len(generated_seqs)

dist_random = calculate_avg_min_levenshtein(random_seqs_str, real_seqs_eval)
dist_baseline = calculate_avg_min_levenshtein(baseline_seqs_str, real_seqs_eval)
dist_rl = calculate_avg_min_levenshtein(rl_tuned_seqs_str, real_seqs_eval)
print(f"  Avg Min Levenshtein (Random vs Real):   {dist_random:.4f}")
print(f"  Avg Min Levenshtein (Baseline vs Real): {dist_baseline:.4f}")
print(f"  Avg Min Levenshtein (RL-Tuned vs Real): {dist_rl:.4f}")

print("\n[Metric 2] 4-mer JSD")
def get_kmer_dist(sequences, k=4):
    all_kmers = [seq[i:i+k] for seq in sequences for i in range(len(seq)-k+1)]
    counts = Counter(all_kmers)
    total = sum(counts.values())
    return {kmer: v/total for kmer, v in counts.items()} if total > 0 else {}

real_kmer_dist = get_kmer_dist(real_seqs_eval)
random_kmer_dist = get_kmer_dist(random_seqs_str)
baseline_kmer_dist = get_kmer_dist(baseline_seqs_str)
rl_kmer_dist = get_kmer_dist(rl_tuned_seqs_str)

all_kmers = sorted(list(set(real_kmer_dist.keys()) | set(baseline_kmer_dist.keys()) | set(rl_kmer_dist.keys()) | set(random_kmer_dist.keys())))
p_real = np.array([real_kmer_dist.get(k, 0) for k in all_kmers])
p_random = np.array([random_kmer_dist.get(k, 0) for k in all_kmers])
p_baseline = np.array([baseline_kmer_dist.get(k, 0) for k in all_kmers])
p_rl = np.array([rl_kmer_dist.get(k, 0) for k in all_kmers])

def jsd(p, q): return jensenshannon(p, q, base=2.0)
print(f"  4-mer JSD (Random vs Real):   {jsd(p_real, p_random):.4f}")
print(f"  4-mer JSD (Baseline vs Real): {jsd(p_real, p_baseline):.4f}")
print(f"  4-mer JSD (RL-Tuned vs Real): {jsd(p_real, p_rl):.4f}")

print("\n[Metric 3] GC Content")
def get_gc(sequences): return [(s.count('G')+s.count('C'))/len(s)*100 for s in sequences]
gc_real = get_gc(real_seqs_eval)
gc_random = get_gc(random_seqs_str)
gc_base = get_gc(baseline_seqs_str)
gc_rl = get_gc(rl_tuned_seqs_str)
plt.figure(figsize=(10,6))
sns.kdeplot(gc_real, label='Real', fill=True)
sns.kdeplot(gc_random, label='Random', fill=True)
sns.kdeplot(gc_base, label='Baseline', fill=True)
sns.kdeplot(gc_rl, label='RL-Tuned', fill=True)
plt.title('GC Content Distribution')
plt.legend()
plt.savefig(os.path.join(output_dir, "gc_content_distribution.png"), dpi=300)
plt.show()

print("\n[Metric 4] t-SNE Visualization")
def get_kmer_features(sequences, k=3):
    # --- FIX: Rename loop variable 'k' to 'kmer' ---
    kmers_vocab = sorted(list(get_kmer_dist(real_seq_strings[:N_EVAL_SAMPLES], k=k).keys()))
    kmer_to_idx = {kmer: i for i, kmer in enumerate(kmers_vocab)}
    features = np.zeros((len(sequences), len(kmer_to_idx)))
    for i, seq in enumerate(sequences):
        dist = get_kmer_dist([seq], k=k)
        for kmer, freq in dist.items(): # Fixed here
            if kmer in kmer_to_idx: features[i, kmer_to_idx[kmer]] = freq
    return features

all_seqs = real_seqs_eval + random_seqs_str + baseline_seqs_str + rl_tuned_seqs_str
labels = ['Real']*len(real_seqs_eval) + ['Random']*len(random_seqs_str) + ['Baseline']*len(baseline_seqs_str) + ['RL-Tuned']*len(rl_tuned_seqs_str)
feats = get_kmer_features(all_seqs)
tsne = TSNE(n_components=2, verbose=0, perplexity=min(30, len(feats)-1), n_iter=1000, random_state=42)
res = tsne.fit_transform(feats)
df_tsne = pd.DataFrame({'x': res[:,0], 'y': res[:,1], 'label': labels})
plt.figure(figsize=(10, 8))
sns.scatterplot(x="x", y="y", hue="label", data=df_tsne, s=50, alpha=0.7)
plt.title('t-SNE Visualization')
plt.savefig(os.path.join(output_dir, "tsne_visualization.png"), dpi=300)
plt.show()

print("\n[Metric 5] FID")
@torch.no_grad()
def get_emb(seqs):
    real_oracle.eval()
    embs = []
    for i in range(0, seqs.shape[0], 32):
        batch = seqs[i:i+32].to(device)
        embs.append(real_oracle.get_embedding(batch).cpu().numpy())
    return np.concatenate(embs, axis=0)
def calc_fid(a, b):
    mu1, s1 = a.mean(0), np.cov(a, rowvar=False)
    mu2, s2 = b.mean(0), np.cov(b, rowvar=False)
    ssdiff = np.sum((mu1-mu2)**2)
    covmean = sqrtm(s1.dot(s2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return ssdiff + np.trace(s1 + s2 - 2*covmean)

real_emb = get_emb(real_one_hot_eval)
rand_emb = get_emb(random_seqs_one_hot)
base_emb = get_emb(baseline_seqs_one_hot)
rl_emb = get_emb(rl_tuned_seqs_one_hot)

print(f"  FID (Random vs Real):   {calc_fid(real_emb, rand_emb):.4f}")
print(f"  FID (Baseline vs Real): {calc_fid(real_emb, base_emb):.4f}")
print(f"  FID (RL-Tuned vs Real): {calc_fid(real_emb, rl_emb):.4f}")

print("\n--- Final Scores (Higher is Better) ---")
print(f"  Random: {rand_preds.mean():.4f}")
print(f"  Baseline: {base_preds.mean():.4f}")
print(f"  RL-Tuned: {rl_preds.mean():.4f}")