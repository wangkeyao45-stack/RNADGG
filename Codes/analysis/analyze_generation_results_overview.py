import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置绘图风格
sns.set_theme(style="whitegrid")
results_dir = "hyperparam_results_final"
csv_path = os.path.join(results_dir, "training_metrics.csv")

if not os.path.exists(csv_path):
    print(f"错误: 未找到 {csv_path}")
else:
    # 1. 读取数据
    df = pd.read_csv(csv_path)

    # 为了避免 style 引起的冲突，我们将 LR 和 Schedule 合并为一个标签
    df['Config'] = df['LR'].astype(str) + " (" + df['Schedule'] + ")"

    # 2. 使用 relplot 代替 FacetGrid 手动映射，这是 Seaborn 最稳定的多图绘制方式
    g = sns.relplot(
        data=df,
        x="Epoch", y="Loss",
        hue="LR",
        style="Schedule",
        col="T",
        row="Channels",
        kind="line",
        facet_kws={'sharey': True, 'margin_titles': True},
        height=4, aspect=1.2
    )

    # 设置 y 轴为对数坐标
    # 在 26 万条数据训练中，不同学习率的 Loss 量级差异巨大，必须使用对数轴
    g.set(yscale="log")

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Comprehensive Hyperparameter Search Analysis (260k Data)")

    # 保存结果图
    save_path = os.path.join(results_dir, "comprehensive_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f">> 综合对比图已保存至: {save_path}")

    # 3. 筛选最优参数组合排名
    final_epoch = df['Epoch'].max()
    final_loss = df[df['Epoch'] == final_epoch].sort_values('Loss')
    print(f"\n=== 最优参数组合排名 (基于第 {final_epoch} 轮) ===")
    print(final_loss[['T', 'Channels', 'LR', 'Schedule', 'Loss']].head(10))

    plt.show()