# -*- coding: utf-8 -*-
# 文件名: search_diffusion_hyperparams_v1.py
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
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import gc
# ==============================================================================
# 关键：从主模型文件导入配置和类
# ==============================================================================
try:
    print("正在导入 experiment_rbs_diffusion_ga_comparison (这可能需要几秒钟来初始化Oracle)...")
    from experiment_rbs_diffusion_ga_comparison import (
        UNet, OracleCNN,
        device, NUCLEOTIDES, SEQUENCE_LENGTH,
        one_hot_sequences, real_oracle  # 导入数据和训练好的Oracle
    )

    print(">> experiment_rbs_diffusion_ga_comparison 导入成功！")
except ImportError as e:
    print("\n[错误] 无法导入 experiment_rbs_diffusion_ga_comparison.py")
    print("请确保该文件与本脚本在同一目录下。")
    print(f"详细错误: {e}")
    exit()


# ==============================================================================
# 1. 增强版 Diffusion (支持不同的噪声调度 Schedule)
# ==============================================================================
class AdvancedDiffusion:
    def __init__(self, model, T=500, schedule='linear'):
        self.model = model
        self.T = T
        self.device = device

        # [实验5核心] 噪声调度策略实现
        if schedule == 'linear':
            # 线性调度: 标准做法
            self.betas = torch.linspace(1e-4, 0.02, T, device=self.device)
        elif schedule == 'cosine':
            # 余弦调度 (Nichol & Dhariwal 2021): 改善信息破坏过程
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
# 2. 实验管理器 (负责调度实验和记录数据)
# ==============================================================================
class ExperimentRunner:
    def __init__(self, output_dir="results_hyperparam_final3"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"实验结果将保存至: {output_dir}/")

    # --- [实验 1 & 3 & 5] 训练阶段 ---
    def run_training_experiment(self, train_loader, lr_list, schedule_list, epochs):
        print(f"\n=== 开始训练实验 (LR, Schedule, Epochs) ===")
        loss_history = []
        overall_best_loss = float('inf')
        best_model_path = None

        for schedule in schedule_list:
            for lr in lr_list:
                # 显存清理
                if 'model' in locals(): del model
                gc.collect()
                torch.cuda.empty_cache()

                exp_name = f"Sched-{schedule}_LR-{lr}"
                print(f"\n正在训练: {exp_name}")

                model = UNet().to(device)
                diff = AdvancedDiffusion(model, T=500, schedule=schedule)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # 增加轻微权重衰减

                current_exp_best_loss = float('inf')

                epoch_pbar = tqdm(range(epochs), desc=f"{exp_name}", leave=True)
                for epoch in epoch_pbar:
                    model.train()
                    batch_losses = []
                    for (x,) in train_loader:
                        optimizer.zero_grad()
                        loss = diff.train_step(x.to(device))

                        # [新增] 数值安全检查：如果爆炸则跳过
                        if torch.isnan(loss):
                            print(f"检测到 NaN，终止当前超参组合: {exp_name}")
                            break

                        loss.backward()
                        # [关键修改] 梯度剪切：解决你第一张图中的数值爆炸问题
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        batch_losses.append(loss.item())

                    if not batch_losses: break  # 出现 NaN 跳过

                    avg_loss = np.mean(batch_losses)
                    loss_history.append({"Epoch": epoch + 1, "Loss": avg_loss, "LR": str(lr), "Schedule": schedule})
                    epoch_pbar.set_postfix(loss=f"{avg_loss:.5f}")

                    # [新增] 追踪全局最优模型用于后续推理
                    if avg_loss < overall_best_loss:
                        overall_best_loss = avg_loss
                        best_model_path = os.path.join(self.output_dir, "best_overall_model.pth")
                        torch.save(model.state_dict(), best_model_path)

                # 保存当前组合的最终权重
                torch.save(model.state_dict(), os.path.join(self.output_dir, f"model_{exp_name}.pth"))

        df_loss = pd.DataFrame(loss_history)
        df_loss.to_csv(os.path.join(self.output_dir, "training_metrics.csv"), index=False)
        print(f">> 实验结束。全局最优 Loss: {overall_best_loss:.6f}, 已保存至: {best_model_path}")
        return best_model_path

    # --- [实验 2 & 3 & 4] 推理阶段 ---
    def run_inference_experiment(self, model_path, oracle_model, guidance_scales, time_steps_list):
        print(f"\n=== 开始推理实验 (Guidance, TimeSteps, Gradients) ===")
        print(f"加载模型: {model_path}")

        model = UNet().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # 假设推理时使用标准的 Linear schedule, T=500 基准
        diff = AdvancedDiffusion(model, T=500, schedule='linear')

        results = []
        grad_dynamics = []

        total_runs = len(time_steps_list) * len(guidance_scales)
        pbar = tqdm(total=total_runs, desc="推理组合测试")

        for T_steps in time_steps_list:
            # [实验3核心] 模拟不同的采样步数
            # 训练是 T=500，如果要测试 T=50，则 stride=10 (每10步采样一次)
            stride = 500 // T_steps

            for g in guidance_scales:
                # 执行采样
                _, mean_score, diversity, grads = self.sample_with_metrics(
                    diff, oracle_model, g, stride, track_grads=True
                )

                results.append({
                    "Guidance": g,
                    "TimeSteps": T_steps,
                    "Oracle_Score": mean_score,
                    "Diversity": diversity
                })

                # [实验4核心] 记录梯度动态
                # 为了避免CSV文件过大，只记录全长采样(T=500)时的梯度详情
                if T_steps == 500:
                    for step_idx, grad_norm in enumerate(grads):
                        # step_idx 是采样循环的次数，对应真实的扩散时间 T -> 0
                        real_t = 500 - (step_idx * stride)
                        grad_dynamics.append({
                            "Guidance": g,
                            "Step_Index": step_idx,
                            "Real_T": real_t,
                            "GradientNorm": grad_norm
                        })

                pbar.update(1)
        pbar.close()

        df_res = pd.DataFrame(results)
        df_grads = pd.DataFrame(grad_dynamics)
        df_res.to_csv(os.path.join(self.output_dir, "inference_metrics.csv"), index=False)
        df_grads.to_csv(os.path.join(self.output_dir, "gradient_dynamics.csv"), index=False)
        print(">> 推理实验数据已保存。")

    # 采样核心逻辑
    def sample_with_metrics(self, diffusion, oracle, guidance, stride, track_grads):
        batch_size = 32  # 测试用 Batch
        # 初始化噪声
        x = torch.randn((batch_size, NUCLEOTIDES, SEQUENCE_LENGTH), device=device)
        grads_record = []

        # 生成倒序时间步列表 [499, ..., 0] 或带 stride
        steps = list(range(0, diffusion.T, stride))[::-1]

        for t_idx in steps:
            t_tensor = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)

            # 1. 计算引导梯度
            grad = torch.zeros_like(x)
            if guidance > 0:
                with torch.enable_grad():
                    x.requires_grad_()
                    s = oracle(x).sum()
                    g_val = torch.autograd.grad(s, x)[0]
                    if track_grads:
                        grads_record.append(g_val.norm().item() / batch_size)
                    grad = g_val.clamp(-1, 1)  # 截断防止梯度爆炸
            elif track_grads:
                grads_record.append(0)

            # 2. 扩散去噪
            with torch.no_grad():
                pred_noise = diffusion.model(x, t_tensor.float())

                alpha_t = diffusion.alphas[t_idx]
                alpha_cum = diffusion.alphas_cumprod[t_idx]

                # 应用 Classifier-Guided Diffusion 公式
                guided_noise = pred_noise - torch.sqrt(1. - alpha_cum) * grad * guidance

                x_mean = (x - (1 - alpha_t) / torch.sqrt(1 - alpha_cum) * guided_noise) / torch.sqrt(alpha_t)

                if t_idx > 0:
                    # 添加随机噪声 (Langevin dynamics part)
                    # 注意：严格来说应该用 posterior variance，这里简化使用 beta
                    noise = torch.randn_like(x)
                    x = x_mean + torch.sqrt(diffusion.betas[t_idx]) * noise
                else:
                    x = x_mean

        # 3. 计算最终指标
        with torch.no_grad():
            final_seqs_oh = F.one_hot(torch.argmax(x, dim=1), NUCLEOTIDES).float().permute(0, 2, 1)
            # 功能性指标 (Functionality)
            scores = oracle(final_seqs_oh).detach().cpu().numpy().mean()

        # 多样性指标 (Diversity - Unique Sequences Ratio)
        seq_strs = ["".join([str(i) for i in s]) for s in torch.argmax(x, dim=1).cpu().numpy()]
        diversity = len(set(seq_strs)) / len(seq_strs)

        return final_seqs_oh, scores, diversity, grads_record


# ==============================================================================
# 3. 自动绘图模块
# ==============================================================================
def plot_results(output_dir):
    sns.set_style("whitegrid")
    print("\n=== 正在生成分析图表 ===")

    # [图表 1] 训练损失对比 (LR & Schedule)
    try:
        df_train = pd.read_csv(os.path.join(output_dir, "training_metrics.csv"))
        plt.figure(figsize=(10, 6))
        # Hue区分LR, Style区分Schedule
        sns.lineplot(data=df_train, x="Epoch", y="Loss", hue="LR", style="Schedule", palette="tab10", linewidth=1.5)
        plt.title("Impact of Learning Rate & Noise Schedule on Convergence")
        plt.yscale('log')  # 使用对数坐标看Loss更清晰
        plt.ylabel("MSE Loss (Log Scale)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "exp_1_5_training_loss.png"), dpi=300)
        plt.close()
        print(f"-> 已生成: exp_1_5_training_loss.png")
    except Exception as e:
        print(f"图表1生成失败: {e}")

    # 读取推理数据
    try:
        df_inf = pd.read_csv(os.path.join(output_dir, "inference_metrics.csv"))

        # [图表 2] Guidance Scale 权衡 (Functionality vs Diversity)
        # 仅选取 T=500 的数据做标准展示
        df_g = df_inf[df_inf['TimeSteps'] == 500]
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Guidance Scale (G)')
        ax1.set_ylabel('Oracle Score (Functionality)', color=color, fontweight='bold')
        sns.lineplot(data=df_g, x='Guidance', y='Oracle_Score', ax=ax1, marker='o', color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(False)

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Diversity (Unique Seq Ratio)', color=color, fontweight='bold')
        sns.lineplot(data=df_g, x='Guidance', y='Diversity', ax=ax2, marker='s', color=color, linewidth=2,
                     linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(True, alpha=0.3)

        plt.title("Trade-off Analysis: Functionality vs. Diversity")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "exp_2_guidance_tradeoff.png"), dpi=300)
        plt.close()
        print(f"-> 已生成: exp_2_guidance_tradeoff.png")

        # [图表 3] 采样步数 (T) 的影响
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_inf, x="TimeSteps", y="Oracle_Score", hue="Guidance", palette="viridis", marker="o")
        plt.title("Effect of Sampling Time Steps (T) on Quality")
        plt.xlabel("Sampling Steps (T)")
        plt.ylabel("Oracle Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "exp_3_timesteps_impact.png"), dpi=300)
        plt.close()
        print(f"-> 已生成: exp_3_timesteps_impact.png")

    except Exception as e:
        print(f"图表2/3生成失败: {e}")

    # [图表 4] 梯度动态分析
    try:
        df_grad = pd.read_csv(os.path.join(output_dir, "gradient_dynamics.csv"))
        plt.figure(figsize=(10, 5))
        # 过滤掉 Guidance=0 的数据(因为梯度为0)
        df_grad_filtered = df_grad[df_grad['Guidance'] > 0]

        sns.lineplot(data=df_grad_filtered, x="Real_T", y="GradientNorm", hue="Guidance", palette="flare",
                     linewidth=1.5)
        plt.title("Gradient Guidance Dynamics during Sampling Process")
        plt.xlabel("Diffusion Time Step (T -> 0)")
        plt.ylabel("Gradient Norm")
        plt.gca().invert_xaxis()  # 反转X轴，符合从噪声(500)到数据(0)的直觉
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "exp_4_gradient_dynamics.png"), dpi=300)
        plt.close()
        print(f"-> 已生成: exp_4_gradient_dynamics.png")
    except Exception as e:
        print(f"图表4生成失败: {e}")


# ==============================================================================
# 4. 主程序入口
# ==============================================================================
if __name__ == "__main__":
    runner = ExperimentRunner("results_hyperparam_final3")

    # 1. 准备训练数据
    print("正在准备数据...")
    # 从 main_model 中获取的 one_hot_sequences
    # 为了演示速度，这里只取前 2000 条数据进行超参搜索训练
    # 正式跑时可以去掉 [:2000]
    train_subset = one_hot_sequences[:260000]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_subset).float()),  # 确保是 float
        batch_size=256,  # 增大 batch_size（原本 64 太小，导致 CPU/GPU 切换过于频繁）
        shuffle=True,
        num_workers=4,  # 开启多进程加载
        pin_memory=True,  # 加速数据从内存拷贝到显存
        drop_last=True  # 避免最后一个不完整的 batch 导致维度波动
    )

    # 2. 执行训练实验 (LR, Schedule)
    # [提示] 为了快速测试，epochs 设为 30。正式实验建议设为 100 或 200
    best_model_path = runner.run_training_experiment(
        train_loader,
        lr_list=[5e-5, 1e-4, 2e-4],
        schedule_list=['linear', 'cosine'],
        epochs=100
    )

    # 3. 执行推理实验 (Guidance, TimeSteps)
    if best_model_path and os.path.exists(best_model_path):
        runner.run_inference_experiment(
            best_model_path,
            real_oracle,
            guidance_scales=[0, 0.5, 1.0, 2.0],  # 探索这几个 guidance 值
            time_steps_list=[50, 100, 250, 500]  # 探索这几个采样步数
        )
    else:
        print("[警告] 未找到模型文件，跳过推理实验。")

    # 4. 生成图表
    plot_results(runner.output_dir)

    print(f"\n所有任务完成！结果已保存在 '{runner.output_dir}' 文件夹中。")