import torch
import os
from experiment_rbs_diffusion_ga_variant3 import (
    UNet, real_oracle, generate_sequences_in_batches,
    one_hot_to_strings, X_test, device, output_dir,
    calculate_novelty, str_train
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 加载最优模型
best_model_path = "hyperparam_results_rigorous/best_rigorous_model.pth" # 确保路径正确
unet = UNet(n_channels=128).to(device)
unet.load_state_dict(torch.load(best_model_path))
unet.eval()

print(">> 开始最终严谨性评估...")

# 2. 采样与指标计算
N_EVAL = 500
gen_oh = generate_sequences_in_batches(unet, real_oracle, N_EVAL, 64, guidance=1.0) #
gen_str = one_hot_to_strings(gen_oh)

nov_score = calculate_novelty(gen_str, str_train) #
with torch.no_grad():
    test_scores = real_oracle(torch.from_numpy(X_test[:N_EVAL]).to(device)).cpu().numpy().flatten()
    gen_scores = real_oracle(gen_oh).cpu().numpy().flatten()

# 3. 强制绘图并保存
plt.figure(figsize=(8, 6))
sns.violinplot(data=[test_scores, gen_scores], palette="pastel")
plt.xticks([0, 1], ["Experimental (Test)", "Diffusion (Generated)"])
plt.title(f"Rigorous Evaluation (Novelty: {nov_score:.2f})")

# 确保保存路径存在
save_path = os.path.join(output_dir, "rigorous_comparison_final.png")
plt.savefig(save_path, dpi=300)
print(f">> 图片已保存至: {save_path}")