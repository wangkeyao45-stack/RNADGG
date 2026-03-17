# -*- coding: utf-8 -*-
# 文件名: oracle_hyperparam_search.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import gc
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# 从您最严谨的模型文件导入数据划分逻辑
try:
    from main_model3 import X_train, X_val, y_train, y_val, device, SEQUENCE_LENGTH, NUCLEOTIDES, set_seed

    print(">> 成功从 main_model3 导入训练与验证集数据。")
except ImportError:
    print(">> 错误：请确保 main_model3.py 在当前目录下且包含划分好的数据。")
    exit()


# ==============================================================================
# 1. 可配置的 Oracle 架构
# ==============================================================================
class FlexibleOracle(nn.Module):
    def __init__(self, n_channels=64, dropout_rate=0.2):
        super(FlexibleOracle, self).__init__()
        # 第一层卷积：捕捉局部碱基模式
        self.conv1 = nn.Conv1d(NUCLEOTIDES, n_channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.pool1 = nn.MaxPool1d(2)

        # 第二层卷积：更高阶的基序组合
        self.conv2 = nn.Conv1d(n_channels, n_channels * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(n_channels * 2)
        self.pool2 = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(dropout_rate)

        # 计算全连接层输入维度
        # 17 -> (pool1) 8 -> (pool2) 4
        flattened_size = (n_channels * 2) * (SEQUENCE_LENGTH // 4)

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        return self.fc(self.dropout(x))


# ==============================================================================
# 2. Oracle 实验管理器
# ==============================================================================
class OracleSearchRunner:
    def __init__(self, output_dir="oracle_search_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics_path = os.path.join(output_dir, "oracle_metrics.csv")

    def run_search(self, train_loader, val_loader, ch_list, lr_list, dr_list, epochs=30):
        results = []
        best_val_mse = float('inf')

        total_exps = len(ch_list) * len(lr_list) * len(dr_list)
        curr = 0

        for ch in ch_list:
            for lr in lr_list:
                for dr in dr_list:
                    curr += 1
                    exp_name = f"CH{ch}_LR{lr}_DR{dr}"
                    print(f"\n[{curr}/{total_exps}] 正在评估 Oracle 配置: {exp_name}")

                    set_seed(42)
                    model = FlexibleOracle(n_channels=ch, dropout_rate=dr).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.MSELoss()

                    for epoch in range(epochs):
                        model.train()
                        train_losses = []
                        for seqs, labels in train_loader:
                            optimizer.zero_grad()
                            preds = model(seqs.to(device)).squeeze()
                            loss = criterion(preds, labels.to(device))
                            loss.backward()
                            optimizer.step()
                            train_losses.append(loss.item())

                        # 验证性能
                        model.eval()
                        val_losses = []
                        with torch.no_grad():
                            for v_seqs, v_labels in val_loader:
                                v_preds = model(v_seqs.to(device)).squeeze()
                                v_loss = criterion(v_preds, v_labels.to(device))
                                val_losses.append(v_loss.item())

                        avg_train = np.mean(train_losses)
                        avg_val = np.mean(val_losses)

                        results.append({
                            "Channels": ch, "LR": lr, "Dropout": dr,
                            "Epoch": epoch + 1, "Train_MSE": avg_train, "Val_MSE": avg_val
                        })

                        if avg_val < best_val_mse:
                            best_val_mse = avg_val
                            torch.save(model.state_dict(), os.path.join(self.output_dir, "best_oracle_final.pth"))

                    del model;
                    gc.collect();
                    torch.cuda.empty_cache()

        pd.DataFrame(results).to_csv(self.metrics_path, index=False)
        print(f"\n>> 搜索完成！结果保存在 {self.metrics_path}")


# ==============================================================================
# 3. 启动搜索
# ==============================================================================
if __name__ == "__main__":
    # 使用 26 万条数据划分出的 Loaders
    t_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=256,
                          shuffle=True)
    v_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=256,
                          shuffle=False)

    runner = OracleSearchRunner()
    runner.run_search(
        t_loader, v_loader,
        ch_list=[64, 128],  # 测试基础宽度
        lr_list=[1e-3, 5e-4],  # 测试学习率
        dr_list=[0.1, 0.3],  # 测试正则化强度以防止过拟合
        epochs=30
    )