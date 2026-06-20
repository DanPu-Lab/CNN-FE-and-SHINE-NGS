import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import numpy as np

input_file = '../finalFE/FE_ION_INDEL_Train.csv'
df = pd.read_csv(input_file)
# 将 'Class' 列作为标签，其余作为特征
X = df.iloc[:, 1:].values  # 特征数据
y = df['Class'].apply(lambda x: 1 if x == 'T' else 0).values  # 将标签转换为二进制

# 使用 LightGBM 进行特征重要性计算
lgb_train = lgb.Dataset(X, y)

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}

lgbm_model = lgb.train(lgb_params, lgb_train, num_boost_round=100)
lgb_importance = lgbm_model.feature_importance()
lgb_feature_names = df.columns[1:]

# 将 LightGBM 特征重要性归一化到 [0, 1]
lgb_importance = lgb_importance / np.max(lgb_importance)

# 输出 LightGBM 重要性大于 0.001 的特征
lgb_important_features = [(name, imp) for name, imp in zip(lgb_feature_names, lgb_importance) if imp > 0.001]
lgb_important_features_sorted = sorted(lgb_important_features, key=lambda x: x[1], reverse=True)

# 输出结果
print("LightGBM 筛选的重要特征:")
for feature, importance in lgb_important_features_sorted:
    print(f"{feature}: {importance}")

# #特征的数量
# b = len(lgb_important_features_sorted)
# print(b)
#
# # 原始的10个特征列表
# original_features = ['BaseQRankSum', 'ReadPosRankSum', 'DP', 'FS', 'MQ', 'MQRankSum', 'QD', 'SOR', 'QUAL', 'GQ']
#
# # 检查在 LightGBM 筛选的特征中是否包含这10个特征
# lgb_selected_originals = [feature for feature, _ in lgb_important_features_sorted if feature in original_features]
#
# # 输出结果
# print("\nLightGBM 筛选的重要特征中包含的原始特征:")
# print(lgb_selected_originals)
#
# # 提取特征列名称
# lgb_important_features_sorted = [feature for feature, _ in lgb_important_features_sorted]
#
# input_file1 = '../finalFE/FE_ION_INDEL_Valid.csv'
# df1 = pd.read_csv(input_file1)
#
# # 创建新 DataFrame，只包含这些特征和第一个标签列
# df_new = df1[['Class'] + lgb_important_features_sorted]
#
# # 保存到新文件
# output_file = 'MF1_ION_INDEL_Valid.csv'
# df_new.to_csv(output_file, index=False)
#
# print(f"新文件已保存到 {output_file}")