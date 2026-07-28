import pandas as pd


def process_csv(input_file, output_file):
    try:
        # 1. 读取 CSV 文件
        # encoding='utf-8' 是常用的编码格式，如果你的文件有中文乱码，可以尝试 'gbk'
        df = pd.read_csv(input_file, encoding='utf-8')

        # 检查必要的列是否存在
        if 'ON' not in df.columns or 'OFF' not in df.columns:
            print("错误：输入文件中缺少 'ON' 或 'OFF' 列。")
            return

        # 2. 计算 ON/OFF 值并存入新列 'ON_OFF_Ratio'
        # 注意：如果 OFF 为 0，结果会是 inf (无穷大)，可以通过代码处理这种情况
        df['ON_OFF_Ratio'] = df.apply(lambda row: row['ON'] / row['OFF'] if row['OFF'] != 0 else 0, axis=1)

        # 3. 保存到新的 CSV 文件
        df.to_csv(output_file, index=False, encoding='utf-8')

        print(f"处理完成！文件已保存为: {output_file}")
        print(f"共处理了 {len(df)} 条数据。")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}，请检查文件名和路径。")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    # --- 配置区域 ---
    # 将这里修改为你实际的文件名
    input_csv_name = 'toehold_data.csv'  # 你的原始文件名
    output_csv_name = 'toehold_data_f.csv'  # 你想生成的输出文件名
    # ----------------

    process_csv(input_csv_name, output_csv_name)