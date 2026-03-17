import pandas as pd
import os


def count_rows_in_csv(file_path):
    """
    读取一个 CSV 文件并返回序列（行）的数量，不包括标题行。

    参数:
    file_path (str): CSV 文件的路径。

    返回:
    int: 文件的行数（数据序列数量），如果发生错误则返回 None。
    """

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在。")
        return None

    try:
        # 1. 使用 pandas 读取 CSV 文件
        # header=0 默认将第一行作为列名
        df = pd.read_csv(file_path)

        # 2. 获取行数。df.shape 是一个 (行数, 列数) 的元组，所以 [0] 是行数。
        num_rows = df.shape[0]

        # 也可以使用 len(df)
        # num_rows = len(df)

        print(f"文件 '{file_path}' 中的序列数量（行数，不含标题）是: {num_rows}")
        return num_rows

    except pd.errors.EmptyDataError:
        print(f"错误: 文件 '{file_path}' 是空的或只有标题行。")
        return 0
    except Exception as e:
        print(f"发生了一个错误: {e}")
        return None

# =========================================================
# 如何使用：
# 将 'your_file.csv' 替换为您实际的文件名
# =========================================================
file_name = '/home/xy_wky/human_5utr_modeling-master/data_pre/toehold_data.csv'
count_rows_in_csv(file_name)

# 示例:
# count_rows_in_csv('data.csv')