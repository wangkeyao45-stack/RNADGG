import pandas as pd
import sys
import os

# --- 配置 ---
# 替换成你实际的CSV文件名
input_filename = 'GSM3130443_designed_library.csv'
# 输出文件名
output_filename = 'utr_data.csv'

# --- 核心处理逻辑 ---

try:
    # 1. 读取CSV文件
    # 假设你的文件是以逗号分隔，且第一行是标题行 (Header=0)。
    # 由于你提供的示例在第一列有一个空值，pandas可能会自动处理或将第一列作为索引。
    df = pd.read_csv(input_filename)

    # 打印原始文件的列名，以便核对
    print("原始文件的列名 (用于确认索引):")
    print(list(df.columns))

    # 根据你提供的列名和数据结构，我们确定要提取的列索引：
    # 'utr' (序列信息) 位于原始数据的第 2 列，对应 DataFrame 的第 1 列 (索引 1)。
    # 'rl' 位于原始数据的第 32 列，对应 DataFrame 的第 31 列 (索引 31)。
    # 注意：如果你的 CSV 文件结构有细微不同，你需要根据上面打印的列名列表来调整索引。

    # 2. 统计数据的数量（行数）
    data_count = len(df)
    print(f"\n✅ 统计结果：总共包含 {data_count} 条数据记录。")


    sequence_index = 1
    rl_index = 32

    new_df = df.iloc[:, [sequence_index, rl_index]].copy()

    # 4. 重命名列以便于理解
    new_df.columns = ['序列', 'rl']

    # 5. 将结果保存到新的CSV文件
    new_df.to_csv(output_filename, index=False, encoding='utf-8')  # index=False表示不写入行索引

    print(f"✅ 提取结果已保存到文件： **{output_filename}**")
    print("\n新文件的前几行数据:")
    print(new_df.head())

except FileNotFoundError:
    print(f"错误：未能找到文件 '{input_filename}'。请确保文件路径和文件名正确无误。")
    sys.exit(1)
except IndexError:
    print("错误：列索引超出范围。请检查您的 CSV 文件是否至少有 32 列，并调整 'sequence_index' 和 'rl_index' 的值。")
    print("当前尝试的索引是 1 (序列) 和 31 (rl)。")
    sys.exit(1)
except Exception as e:
    print(f"处理文件时发生未预期的错误: {e}")
    sys.exit(1)