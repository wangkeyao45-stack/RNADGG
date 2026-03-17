import pandas as pd
import os


def count_data_rows_with_pandas(file_path):
    """
    使用 pandas 统计 CSV 文件中除去表头后的数据行数。

    参数:
    file_path (str): CSV 文件的路径。

    返回:
    int: 文件的有效数据行数（不含表头），如果发生错误则返回 None。
    """

    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在。")
        return None

    try:
        # 1. 使用 pandas 读取 CSV 文件
        # pandas 默认将第一行作为列名/表头
        df = pd.read_csv(file_path)

        # 2. 获取 DataFrame 的行数
        # df.shape[0] 返回行数，即数据序列的数量（不包含已作为表头的首行）
        num_rows = df.shape[0]

        # 也可以使用 len(df)
        # num_rows = len(df)

        print(f"文件 '{file_path}' 中有效数据序列数量（不含表头）是: {num_rows}")
        return num_rows

    except pd.errors.EmptyDataError:
        print(f"警告: 文件 '{file_path}' 是空的或只包含表头。")
        return 0
    except Exception as e:
        print(f"发生了一个错误: {e}")
        return None

# =========================================================
# 如何使用：
# =========================================================
# 替换为您的文件路径
file_name = '/home/xy_wky/RBS/data/output.csv'
count_data_rows_with_pandas(file_name)