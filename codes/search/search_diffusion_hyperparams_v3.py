# -*- coding: utf-8 -*-
# 文件名: search_diffusion_hyperparams_v3.py
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
# 1. 从严谨版 experiment_rbs_diffusion_ga_variant3 导入组件
# ==============================================================================
try:
    print("正在导入 experiment_rbs_diffusion_ga_variant3 (含严格数据划分)...")
    from experiment_rbs_diffusion_ga_variant3 import (
        UNet, device, NUCLEOTIDES, SEQUENCE_LENGTH,
        X_train, X_val, real_oracle, set_seed
    )

    print(">> experiment_rbs_diffusion_ga_variant3 导入成功！")
except ImportError as e:
    print(f"\n[错误] 无法导入 experiment_rbs_diffusion_ga_variant3.py: {e}")
    exit()


# ==============================================================================
# 2. 增强版 Diffusion
# ==============================================================================
class AdvancedDiffusion:
    def __init__(self, model, T=500, schedule='linear'):
        self.model = model
        self.T = T
        self.device = device
        if schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, T, device=self.device)
        elif schedule == 'cosine':
            s = 0.008
            x = torch.linspace(0, T, T + 1, device=self.device)
            alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def train_step(self, x_0):
        t = torch.randint(0, self.T, (x_0.shape[0],), device=self.device)
        noise = torch.randn_like(x_0)
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1)
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        predicted_noise = self.model(x_t, t.float())
        return F.mse_loss(predicted_noise, noise)


# ==============================================================================
# 3. 严谨实验管理器 (支持双曲线监控与多种子)
# ==============================================================================
class RigorousExperimentRunner:
    def __init__(self, output_dir="results_rigorous_v3"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics_path = os.path.join(output_dir, "rigorous_metrics.csv")

    def run_grid_search(self, train_loader, val_loader, lr_list, schedule_list, T_list, channels_list, seeds=[42]):
        loss_history = []
        overall_best_val_loss = float('inf')

        for seed in seeds:
            set_seed(seed)
            for T_val in T_list:
                for ch in channels_list:
                    for schedule in schedule_list:
                        for lr in lr_list:
                            exp_name = f"S{seed}_T{T_val}_CH{ch}_{schedule}_LR{lr}"
                            print(f"\n正在执行实验: {exp_name}")

                            if 'model' in locals(): del model
                            gc.collect();
                            torch.cuda.empty_cache()

                            model = UNet(n_channels=ch).to(device)
                            diff = AdvancedDiffusion(model, T=T_val, schedule=schedule)
                            optimizer = optim.Adam(model.parameters(), lr=lr)

                            pbar = tqdm(range(50), desc="Epochs")
                            for epoch in pbar:
                                # --- 训练阶段 ---
                                model.train()
                                train_batch_losses = []
                                for (x,) in train_loader:
                                    optimizer.zero_grad()
                                    loss = diff.train_step(x.to(device))
                                    loss.backward()
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度剪切消融默认开启
                                    optimizer.step()
                                    train_batch_losses.append(loss.item())

                                # --- 验证阶段 (严谨性核心) ---
                                model.eval()
                                val_batch_losses = []
                                with torch.no_grad():
                                    for (x_v,) in val_loader:
                                        val_loss = diff.train_step(x_v.to(device))
                                        val_batch_losses.append(val_loss.item())

                                avg_train = np.mean(train_batch_losses)
                                avg_val = np.mean(val_batch_losses)

                                loss_history.append({
                                    "Seed": seed, "T": T_val, "Channels": ch, "Schedule": schedule, "LR": lr,
                                    "Epoch": epoch + 1, "Train_Loss": avg_train, "Val_Loss": avg_val
                                })
                                pbar.set_postfix(T_L=f"{avg_train:.4f}", V_L=f"{avg_val:.4f}")

                                if avg_val < overall_best_val_loss:
                                    overall_best_val_loss = avg_val
                                    torch.save(model.state_dict(),
                                               os.path.join(self.output_dir, "best_rigorous_model.pth"))

        pd.DataFrame(loss_history).to_csv(self.metrics_path, index=False)
        return self.metrics_path


# ==============================================================================
# 4. 主程序入口
# ==============================================================================
if __name__ == "__main__":
    runner = RigorousExperimentRunner("hyperparam_results_rigorous")

    # 利用 main_model3 划分好的 X_train 和 X_val
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float()), batch_size=256, shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float()), batch_size=256, shuffle=False)

    best_metrics = runner.run_grid_search(
        train_loader, val_loader,
        lr_list=[1e-4, 2e-4],
        schedule_list=['linear', 'cosine'],
        T_list=[1000],
        channels_list=[128],
        seeds=[42, 123]  # 使用双种子验证稳定性
    )

    print(f"\n严谨搜索完成。指标已保存至: {best_metrics}")