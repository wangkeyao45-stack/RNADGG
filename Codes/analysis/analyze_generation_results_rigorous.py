import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 强制使用非交互式后端，防止服务器报错
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置绘图风格
sns.set_theme(style="whitegrid")
results_dir = "hyperparam_results_rigorous"
csv_path = os.path.join(results_dir, "rigorous_metrics.csv")

if not os.path.exists(csv_path):
    print(f"错误: 未找到 {csv_path}")
else:
    # 1. 读取数据
    df = pd.read_csv(csv_path)

    # --- 关键修改：合成 Config 列 ---
    # 将多个超参数组合成一个易读的标签用于绘图
    df['Config'] = (
            "T=" + df['T'].astype(str) +
            ", CH=" + df['Channels'].astype(str) +
            ", " + df['Schedule'].astype(str) +
            ", LR=" + df['LR'].astype(str)
    )

    # 2. 绘制泛化差距图 (Generalization Gap)
    # 聚合多个种子并绘制训练与验证曲线
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Epoch", y="Train_Loss", label="Training Loss", linewidth=2)
    sns.lineplot(data=df, x="Epoch", y="Val_Loss", label="Validation Loss", linewidth=2, linestyle='--')
    plt.yscale('log')
    plt.title("Generalization Gap Analysis (260k Dataset)")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "generalization_gap.png"), dpi=300)
    print(f">> 泛化差距图已保存至: {os.path.join(results_dir, 'generalization_gap.png')}")

    # 3. 绘制多随机种子稳定性图
    # 展示不同 Seed 下的最终验证损失，证明结果的可复现性
    plt.figure(figsize=(10, 6))
    final_epoch = df['Epoch'].max()
    final_losses = df[df['Epoch'] == final_epoch]

    sns.barplot(data=final_losses, x="Seed", y="Val_Loss", hue="Config", palette="muted")
    plt.title(f"Model Robustness across Multiple Random Seeds (Epoch {final_epoch})")
    plt.ylabel("Final Validation Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Configurations')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "seed_robustness.png"), dpi=300)
    print(f">> 稳定性对比图已保存至: {os.path.join(results_dir, 'seed_robustness.png')}")

    # 4. 打印性能摘要
    best_config = final_losses.sort_values("Val_Loss").iloc[0]
    print("\n" + "=" * 30)
    print("=== 严谨评估结果摘要 ===")
    print(f"最优验证集损失 (Val Loss): {best_config['Val_Loss']:.6f}")
    print(f"对应超参数: {best_config['Config']}")
    print(f"随机种子 (Seed): {best_config['Seed']}")
    print("=" * 30)