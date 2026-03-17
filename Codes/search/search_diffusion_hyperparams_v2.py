# -*- coding: utf-8 -*-
# 文件名: search_diffusion_hyperparams_v2.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import gc
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# ==============================================================================
# 1. 从修改后的 experiment_rbs_diffusion_ga_variant2 导入组件
# ==============================================================================
try:
    print("正在导入 experiment_rbs_diffusion_ga_variant2 (初始化数据与Oracle)...")
    from experiment_rbs_diffusion_ga_variant2 import (
        UNet, device, NUCLEOTIDES, SEQUENCE_LENGTH,
        one_hot_sequences, real_oracle, idx_to_char
    )

    print(">> experiment_rbs_diffusion_ga_variant2 导入成功！")
except ImportError as e:
    print(f"\n[错误] 无法导入 experiment_rbs_diffusion_ga_variant2.py: {e}")
    exit()


# ==============================================================================
# 2. 增强版 Diffusion (支持动态 T 和 Schedule)
# ==============================================================================
class AdvancedDiffusion:
    def __init__(self, model, T=500, schedule='linear'):
        self.model = model
        self.T = T
        self.device = device

        # 噪声调度策略实现
        if schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, T, device=self.device)
        elif schedule == 'cosine':
            s = 0.008
            steps = T + 1
            x = torch.linspace(0, T, steps, device=self.device)
            alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def q_sample(self, x_0, t, noise=None):
        if noise is None: noise = torch.randn_like(x_0)
        mean = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1) * x_0
        std = torch.sqrt(1. - self.alphas_cumprod[t]).view(-1, 1, 1)
        return mean + std * noise

    def train_step(self, x_0):
        t = torch.randint(0, self.T, (x_0.shape[0],), device=self.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t.float())
        return F.mse_loss(predicted_noise, noise)


# ==============================================================================
# 3. 实验管理器 (多维搜索与自动保存)
# ==============================================================================
class ExperimentRunner:
    def __init__(self, output_dir="results_comprehensive_v2"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics_path = os.path.join(output_dir, "training_metrics.csv")

    def run_grid_search(self, train_loader, lr_list, schedule_list, T_list, channels_list, wd_list, epochs):
        print(f"=== 启动全维度超参探索 (大数据集模式) ===")
        loss_history = []
        overall_best_loss = float('inf')
        best_model_path = None

        # 计算总实验数
        total_exps = len(lr_list) * len(schedule_list) * len(T_list) * len(channels_list) * len(wd_list)
        curr_exp = 0

        for T_val in T_list:
            for ch in channels_list:
                for wd in wd_list:
                    for schedule in schedule_list:
                        for lr in lr_list:
                            curr_exp += 1
                            exp_name = f"T{T_val}_CH{ch}_WD{wd}_{schedule}_LR{lr}"
                            print(f"\n[{curr_exp}/{total_exps}] 正在训练: {exp_name}")

                            # 显存清理逻辑
                            if 'model' in locals(): del model
                            gc.collect()
                            torch.cuda.empty_cache()

                            # 初始化动态通道数的 UNet
                            model = UNet(n_channels=ch).to(device)
                            diff = AdvancedDiffusion(model, T=T_val, schedule=schedule)
                            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                            epoch_pbar = tqdm(range(epochs), desc=f"Training", leave=False)
                            for epoch in epoch_pbar:
                                model.train()
                                batch_losses = []
                                for (x,) in train_loader:
                                    optimizer.zero_grad()
                                    loss = diff.train_step(x.to(device))

                                    if torch.isnan(loss):  # NaN 保护
                                        print(f"检测到 NaN，终止此组合实验")
                                        break

                                    loss.backward()
                                    # 梯度剪切：防止数值爆炸
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                    optimizer.step()
                                    batch_losses.append(loss.item())

                                if not batch_losses: break

                                avg_loss = np.mean(batch_losses)
                                loss_history.append({
                                    "T": T_val, "Channels": ch, "WD": wd,
                                    "Schedule": schedule, "LR": lr, "Loss": avg_loss, "Epoch": epoch + 1
                                })
                                epoch_pbar.set_postfix(loss=f"{avg_loss:.5f}")

                                # 自动保存全局最优模型权重
                                if avg_loss < overall_best_loss:
                                    overall_best_loss = avg_loss
                                    best_model_path = os.path.join(self.output_dir, "best_overall_model.pth")
                                    torch.save(model.state_dict(), best_model_path)

                            # 保存当前组合权重
                            torch.save(model.state_dict(), os.path.join(self.output_dir, f"model_{exp_name}.pth"))

        df = pd.DataFrame(loss_history)
        df.to_csv(self.metrics_path, index=False)
        return best_model_path


# ==============================================================================
# 4. 主程序入口
# ==============================================================================
if __name__ == "__main__":
    runner = ExperimentRunner("hyperparam_results_final")

    # 数据准备 (26万条数据)
    print("正在准备大数据集加载器...")
    train_subset = one_hot_sequences[:260000]
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_subset).float()),
        batch_size=256,  # 如果显存不足(OOM)，调小至 128
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 启动多维探索
    # 注意：实验总数 = 列表长度乘积 (当前配置为 2*2*2*2*1 = 16 组)
    best_path = runner.run_grid_search(
        train_loader,
        lr_list=[1e-4, 2e-4],  # 稳健的学习率
        schedule_list=['linear', 'cosine'],  # 噪声调度
        T_list=[500, 1000],  # 扩散深度
        channels_list=[64, 128],  # 模型宽度 (UNet 通道数)
        wd_list=[1e-5],  # 权重衰减 (正则化)
        epochs=50  # 每个实验跑 50 轮
    )

    print(f"\n所有任务完成！最优模型已保存至: {best_path}")
    print("您现在可以查看 CSV 文件并进行可视化分析。")