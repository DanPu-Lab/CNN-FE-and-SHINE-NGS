import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform

# 读取训练数据（用于计算特征顺序）
input_file = 'MF1_ION_INDEL_Train.csv'
df_train = pd.read_csv(input_file)
features = df_train.columns[1:].tolist()  # 提取特征列名（假设第一列为标签 'Class'）

# 计算 Spearman 相关性矩阵
correlation_matrix = df_train.iloc[:, 1:].corr(method='spearman')

# --- 关键修正：将相关性转换为距离矩阵 ---
# 公式：距离 = 1 - |相关性|，确保负相关也被视为“相近”
distance_matrix = 1 - np.abs(correlation_matrix)

# 将距离矩阵压缩为 linkage 函数需要的格式（上三角展开为一维数组）
condensed_distance = squareform(distance_matrix, checks=False)

# 使用层次聚类计算链接矩阵（基于正确的距离输入）
linkage_matrix = linkage(condensed_distance, method='average')

# 获取特征重排序索引
ordered_indices = leaves_list(linkage_matrix)
ordered_features = [features[i] for i in ordered_indices]  # 按聚类顺序排列特征名

# --- 处理测试数据 ---
input_file1 = 'MF1_ION_INDEL_Test.csv'
df_test = pd.read_csv(input_file1)

# 确保测试数据特征与训练集一致（防止列名或数量不一致导致错误）
# 移除测试数据中可能存在的额外列（若有）
df_test = df_test[['Class'] + features]

# 按聚类后的特征顺序重排测试数据
df_ordered = df_test[['Class'] + ordered_features]

# 输出新的 CSV 文件
output_file = 'MF2_ION_INDEL_Test_xiu.csv'
df_ordered.to_csv(output_file, index=False)
print(f"已按新的顺序输出 CSV 文件到 {output_file}")

# 可选：绘制热力图（需安装 seaborn）
# import seaborn as sns
# sns.clustermap(correlation_matrix.loc[ordered_features, ordered_features],
#                cmap='coolwarm', figsize=(12, 10), vmin=-1, vmax=1)
# plt.title('Ordered Correlation Matrix')
# plt.show()