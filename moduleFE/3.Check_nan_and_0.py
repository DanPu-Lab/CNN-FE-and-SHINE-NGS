import pandas as pd
import numpy as np

# 读取CSV文件（替换为你的文件路径）
df = pd.read_csv("ILM_SNP_Test.csv")

# --------------------------
# 检查缺失值（NaN/None）
# --------------------------
# 获取存在缺失值的行索引
missing_rows = df[df.isna().any(axis=1)]
has_missing = not missing_rows.empty

print("=" * 40)
print("缺失值检查结果:")
if has_missing:
    print(f"共发现 {len(missing_rows)} 行存在缺失值")
    print("具体行号及缺失列:")

    # 逐行显示缺失位置
    for idx, row in missing_rows.iterrows():
        missing_cols = row.index[row.isna()].tolist()
        print(f"行 {idx}: 缺失列 → {missing_cols}")
else:
    print("✓ 未发现缺失值")

# --------------------------
# 检查无穷值（inf/-inf）
# --------------------------
# 筛选数值列
numeric_cols = df.select_dtypes(include=np.number).columns

inf_rows = pd.DataFrame()
if not numeric_cols.empty:
    # 检测无穷值
    inf_mask = df[numeric_cols].applymap(np.isinf)
    has_inf = inf_mask.any().any()

    print("\n" + "=" * 40)
    print("无穷值检查结果:")
    if has_inf:
        # 获取存在无穷值的行索引和列名
        inf_rows = df[inf_mask.any(axis=1)]
        print(f"共发现 {len(inf_rows)} 行存在无穷值")
        print("具体行号及问题列:")

        for idx, row in inf_rows.iterrows():
            inf_cols = numeric_cols[inf_mask.loc[idx]].tolist()
            print(f"行 {idx}: 无穷值列 → {inf_cols}")
    else:
        print("✓ 未发现无穷值")
else:
    print("\n" + "=" * 40)
    print("无穷值检查结果: 无数值列，跳过检查")

# ====================================
# 最终汇总
# ====================================
print("\n" + "=" * 40)
print("最终统计:")
print(f"总缺失行数: {len(missing_rows)}")
if not numeric_cols.empty:
    print(f"总无穷值行数: {len(inf_rows)}")

# 可选：保存问题行到新文件
# problem_rows = pd.concat([missing_rows, inf_rows]).drop_duplicates()
# problem_rows.to_csv("problem_rows.csv", index=False)