import pandas as pd
import numpy as np

# =========================
# 1. 文件路径
# =========================
train_file = 'MF2_ILM_SNP_Train.csv'
valid_file = 'MF2_ILM_SNP_Valid.csv'
test_file  = 'MF2_ILM_SNP_Test.csv'

# =========================
# 2. 读取数据
# =========================
train_df = pd.read_csv(train_file)
valid_df = pd.read_csv(valid_file)
test_df  = pd.read_csv(test_file)

# =========================
# 3. 替换 inf → NaN
# =========================
train_df = train_df.replace([np.inf, -np.inf], np.nan)
valid_df = valid_df.replace([np.inf, -np.inf], np.nan)
test_df  = test_df.replace([np.inf, -np.inf], np.nan)

# =========================
# 4. 找出含 NaN（来自 inf）的列
# =========================
train_inf_cols = set(train_df.columns[train_df.isna().any()])
valid_inf_cols = set(valid_df.columns[valid_df.isna().any()])
test_inf_cols  = set(test_df.columns[test_df.isna().any()])

# =========================
# 5. 统一要删除的列（取并集）
# =========================
drop_cols = list(train_inf_cols | valid_inf_cols | test_inf_cols)

print(f"Found {len(drop_cols)} problematic columns:")
print(drop_cols)

# =========================
# 6. 删除这些列
# =========================
train_df = train_df.drop(columns=drop_cols)
valid_df = valid_df.drop(columns=drop_cols)
test_df  = test_df.drop(columns=drop_cols)

# =========================
# 7. 再次保险处理 NaN
# =========================
train_df = train_df.fillna(0)
valid_df = valid_df.fillna(0)
test_df  = test_df.fillna(0)

# =========================
# 8. 保存
# =========================
train_df.to_csv("MF2_ILM_SNP_Train.csv", index=False)
valid_df.to_csv("MF2_ILM_SNP_Valid.csv", index=False)
test_df.to_csv("MF2_ILM_SNP_Test.csv", index=False)

print("Done: cleaned datasets saved.")