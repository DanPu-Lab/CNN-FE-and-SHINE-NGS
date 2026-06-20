import pandas as pd

# 加载三个 CSV 文件
# file1_path = 'FE_ILM_INDEL_Train.csv'  # 替换为你的文件路径
# file2_path = 'FE_ILM_INDEL_Valid.csv'  # 替换为你的文件路径
# file3_path = 'FE_ILM_INDEL_Test.csv'  # 替换为你的文件路径

# file1_path = 'MF1_ILM_INDEL_Train.csv'  # 替换为你的文件路径
# file2_path = 'MF1_ILM_INDEL_Valid.csv'  # 替换为你的文件路径
# file3_path = 'MF1_ILM_INDEL_Test.csv'  # 替换为你的文件路径

file1_path = 'MF2_ION_SNP_Train.csv'  # 替换为你的文件路径
file2_path = 'MF2_ION_SNP_Valid.csv'  # 替换为你的文件路径
file3_path = 'MF2_ION_SNP_Test.csv'  # 替换为你的文件路径

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)
df3 = pd.read_csv(file3_path)

# 获取每个文件的特征列名称（去掉标签列）
features_df1 = set(df1.columns[1:])
features_df2 = set(df2.columns[1:])
features_df3 = set(df3.columns[1:])

# 检查是否对应
all_features_same = features_df1 == features_df2 == features_df3

if all_features_same:
    print("三个 CSV 文件中的特征行名称一一对应。")
else:
    print("三个 CSV 文件中的特征行名称不完全对应。")

    # 找出不匹配的特征
    print("\n不匹配的特征：")
    print("仅在第一个文件中的特征：", features_df1 - features_df2 - features_df3)
    print("仅在第二个文件中的特征：", features_df2 - features_df1 - features_df3)
    print("仅在第三个文件中的特征：", features_df3 - features_df1 - features_df2)
