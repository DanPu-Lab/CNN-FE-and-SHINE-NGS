import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list


# 读取数据
input_file = 'MF1_ION_INDEL_Train.csv'  # 替换为你的数据路径
df = pd.read_csv(input_file)

# 计算 Spearman 相关性矩阵
correlation_matrix = df.iloc[:, 1:].corr(method='spearman')

# 使用层次聚类计算链接矩阵
linkage_matrix = linkage(correlation_matrix, method='average')
ordered_indices = leaves_list(linkage_matrix)
ordered_features = correlation_matrix.columns[ordered_indices].tolist()

input_file1 = 'MF1_ION_INDEL_Test.csv'
df1 = pd.read_csv(input_file1)
# 按新的顺序排列数据框
df_ordered = df1[['Class'] + ordered_features]  # 确保第一个标签列 'Class' 保持在第一列

# 输出新的 CSV 文件
output_file = 'MF2_ION_INDEL_Test.csv'
df_ordered.to_csv(output_file, index=False)

print(f"已按新的顺序输出 CSV 文件到 {output_file}")

# # 绘制热力图，带有层次聚类
# sns.clustermap(correlation_matrix, cmap='coolwarm', linewidths=0.1, figsize=(18, 16), method='average', vmin=-1, vmax=1, annot=False)
#
# plt.title('Feature Correlation and Hierarchical Clustering')
# plt.show()
