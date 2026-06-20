# 导入必要库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

# -------------------------- 基础设置 --------------------------
# 字体设置（英文显示，适配学术场景）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')  # 图表样式
np.random.seed(42)  # 固定随机种子，保证结果可复现

# -------------------------- 1. 生成模拟数据（含冗余特征） --------------------------
n_samples = 500  # 样本数
n_features = 15  # 特征数（含冗余，模拟非线性相关）
features = pd.DataFrame()

# 生成带非线性冗余的特征（模拟真实场景中的特征相关性）
features['Feature_1'] = np.random.randn(n_samples)
features['Feature_2'] = np.sin(features['Feature_1']) + np.random.randn(n_samples)*0.1  # 与1非线性冗余
features['Feature_3'] = np.cos(features['Feature_1']) + np.random.randn(n_samples)*0.1  # 与1非线性冗余
features['Feature_4'] = np.random.randn(n_samples)
features['Feature_5'] = features['Feature_4']**2 + np.random.randn(n_samples)*0.2  # 与4非线性冗余
features['Feature_6'] = np.exp(-abs(features['Feature_4'])) + np.random.randn(n_samples)*0.1  # 与4非线性冗余
features['Feature_7'] = np.random.randn(n_samples)
features['Feature_8'] = np.random.randn(n_samples)
features['Feature_9'] = np.sin(features['Feature_7'] + features['Feature_8']) + np.random.randn(n_samples)*0.1  # 与7、8相关
features['Feature_10'] = np.random.randn(n_samples)
features['Feature_11'] = features['Feature_10'] * 0.8 + np.random.randn(n_samples)*0.1  # 与10线性冗余
features['Feature_12'] = np.random.randn(n_samples)
features['Feature_13'] = np.cos(features['Feature_12'] * 2) + np.random.randn(n_samples)*0.1
features['Feature_14'] = np.random.randn(n_samples)
features['Feature_15'] = np.sin(features['Feature_14'] * 3) + np.random.randn(n_samples)*0.1

# -------------------------- 2. 核心计算（按你的技术描述实现） --------------------------
# 2.1 计算斯皮尔曼相关系数矩阵（非线性相关性）
corr_matrix, _ = spearmanr(features)
corr_matrix = pd.DataFrame(corr_matrix, columns=features.columns, index=features.columns)

# 2.2 相关性转距离矩阵（消除正负方向差异：distance = 1 - |correlation|）
distance_matrix = 1 - np.abs(corr_matrix)

# 2.3 层次聚类（平均链接法）
condensed_distance = squareform(distance_matrix)  # 转换为压缩矩阵（适配linkage函数）
linkage_matrix = linkage(condensed_distance, method='average')  # 平均链接法

# 2.4 提取聚类结果（用于特征重排和分组）
dendro = dendrogram(linkage_matrix, no_plot=True)
feature_order = [features.columns[i] for i in dendro['leaves']]  # 聚类树叶节点顺序（特征重排依据）
n_clusters = 4  # 聚类簇数（可按需调整）
cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')  # 特征分组标签

# -------------------------- 3. 绘制图1：相关性矩阵 + 层次聚类树 --------------------------
fig, ax = plt.subplots(figsize=(12, 10))
# 整合热力图和聚类树（自动应用层次聚类）
g = sns.clustermap(
    corr_matrix,
    method='average',  # 层次聚类方法（与你的描述一致）
    metric='precomputed',  # 使用预计算的距离矩阵
    row_linkage=linkage_matrix,
    col_linkage=linkage_matrix,
    cmap='RdBu_r',  # 红蓝配色（正相关红、负相关蓝、无相关白）
    center=0,  # 颜色中心为0（相关系数=0）
    annot=True,  # 显示相关系数数值
    fmt='.2f',  # 数值保留2位小数
    annot_kws={'size': 6},  # 数值字体大小
    cbar_kws={'label': 'Spearman Correlation Coefficient'},  # 颜色条标签
    figsize=(12, 10)
)
g.fig.suptitle(
    'Feature Correlation Matrix (Spearman) with Hierarchical Clustering\n(Average Linkage Method)',
    fontsize=16, fontweight='bold', y=1.02  # 标题位置调整
)
# 保存图1（可修改保存路径）
g.savefig('feature_correlation_hierarchical_clustering.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------- 4. 绘制图2：重排后相关性矩阵 + 聚类分组 --------------------------
# 4.1 按聚类结果重排相关性矩阵
rearranged_corr = corr_matrix.reindex(index=feature_order, columns=feature_order)

# 4.2 准备聚类分组颜色标注
cluster_df = pd.DataFrame({'Feature': features.columns, 'Cluster': cluster_labels})
rearranged_clusters = cluster_df.set_index('Feature').loc[feature_order, 'Cluster']
cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 簇颜色（可自定义）
color_map = {i+1: cluster_colors[i] for i in range(n_clusters)}
rearranged_cluster_colors = [color_map[c] for c in rearranged_clusters]

# 4.3 绘制重排后的热力图
fig, (ax_main, ax_side) = plt.subplots(1, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [20, 1]})

# 热力图主体
im = ax_main.imshow(rearranged_corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax_main.set_xticks(range(len(feature_order)))
ax_main.set_yticks(range(len(feature_order)))
ax_main.set_xticklabels(feature_order, rotation=45, ha='right', fontsize=10)
ax_main.set_yticklabels(feature_order, fontsize=10)
ax_main.set_title(
    'Rearranged Feature Correlation Matrix\n(Ordered by Hierarchical Clustering)',
    fontsize=16, fontweight='bold', pad=20
)

# 标注强相关系数（仅显示绝对值>0.3的数值，避免拥挤）
for i in range(len(feature_order)):
    for j in range(len(feature_order)):
        if abs(rearranged_corr.iloc[i, j]) > 0.3:
            ax_main.text(j, i, f'{rearranged_corr.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=7)

# 右侧聚类分组标注
ax_side.barh(range(len(rearranged_cluster_colors)), [1]*len(rearranged_cluster_colors),
             color=rearranged_cluster_colors, edgecolor='black', linewidth=0.5)
ax_side.set_yticks(range(len(feature_order)))
ax_side.set_yticklabels([f'Cluster {c}' for c in rearranged_clusters], fontsize=10)
ax_side.set_xlim(0, 1)
ax_side.set_xticks([])
ax_side.set_title('Feature Groups', fontsize=12, fontweight='bold', pad=10)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax_main, shrink=0.8)
cbar.set_label('Spearman Correlation Coefficient', fontsize=12)

# 调整布局并保存图2
plt.tight_layout()
plt.savefig('rearranged_feature_correlation_with_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

print("两张图已生成完成！")
print("文件1：feature_correlation_hierarchical_clustering.png（相关性矩阵+聚类树）")
print("文件2：rearranged_feature_correlation_with_clusters.png（重排矩阵+分组标注）")