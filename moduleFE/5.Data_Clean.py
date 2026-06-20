import pandas as pd
import numpy as np

def check(df):
    df = df.iloc[:, 1:]

    # 检查无穷值
    cols_with_inf = df.columns[df.isin([np.inf, -np.inf]).any()]
    if not cols_with_inf.empty:
        print("存在无穷值的列名：", cols_with_inf.tolist())
    else:
        print("数据没有无穷值")

    # 检查缺失值
    cols_with_null = df.columns[df.isnull().any()]
    if not cols_with_null.empty:
        print("存在缺失值的列名：", cols_with_null.tolist())
    else:
        print("数据没有缺失值")

    return cols_with_inf, cols_with_null

input_file = 'FE_ION_INDEL_Test.csv'
df = pd.read_csv(input_file)

cols_with_inf, cols_with_null = check(df)

if not cols_with_inf.empty or not cols_with_null.empty:
    cols_with_drop = cols_with_inf.union(cols_with_null)
    df = df.drop(columns=cols_with_drop)
    print("已删除包含无穷值和缺失值的列。")

print("\n删除列后再次检测：")
check(df)

output_file = 'FE_ION_INDEL_Test.csv'
df.to_csv(output_file, index = False)