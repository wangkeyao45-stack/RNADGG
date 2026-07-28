import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置绘图风格
sns.set_theme(style="whitegrid")
results_dir = "oracle_search_results"
csv_path = os.path.join(results_dir, "oracle_metrics.csv")

if not os.path.exists(csv_path):
    print(f"错误: 未找到 {csv_path}")
else:
    df = pd.read_csv(csv_path)

    # 合成配置标签
    df['Config'] = "CH" + df['Channels'].astype(str) + "_LR" + df['LR'].astype(str) + "_DR" + df['Dropout'].astype(str)

    # 1. 绘制验证集 MSE 趋势图 (评估拟合速度与深度)
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x="Epoch", y="Val_MSE", hue="Config", palette="flare", linewidth=2)
    plt.yscale('log')
    plt.title("Oracle Validation MSE Across Different Configurations (260k Data)")
    plt.ylabel("Validation MSE (Log Scale)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Hyperparams")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "oracle_val_mse_trend.png"), dpi=300)

    # 2. 绘制泛化差距图 (训练误差 vs 验证误差)
    # 我们选择最终 Epoch 的数据来展示过拟合程度
    final_epoch = df['Epoch'].max()
    final_df = df[df['Epoch'] == final_epoch].melt(
        id_vars=['Config'], value_vars=['Train_MSE', 'Val_MSE'],
        var_name='Type', value_name='MSE'
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=final_df, x="Config", y="MSE", hue="Type", palette="muted")
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Generalization Gap Analysis at Epoch {final_epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "oracle_generalization_gap.png"), dpi=300)

    # 3. 打印最优排名
    best_oracle = df[df['Epoch'] == final_epoch].sort_values("Val_MSE").iloc[0]
    print("\n" + "=" * 40)
    print("=== Oracle 搜索最终结论 ===")
    print(f"最优配置: {best_oracle['Config']}")
    print(f"最小验证 MSE: {best_oracle['Val_MSE']:.6f}")
    print(f"训练 MSE: {best_oracle['Train_MSE']:.6f}")
    print("=" * 40)