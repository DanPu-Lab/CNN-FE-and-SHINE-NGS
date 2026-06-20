import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 替换为你的实际文件路径（前5个为模型A，后5个为模型B）
model_a_files = [
    "Resnet34_ION_INDEL1.csv", "Resnet34_ION_INDEL2.csv", "Resnet34_ION_INDEL3.csv", "Resnet34_ION_INDEL4.csv", "Resnet34_ION_INDEL5.csv"
]
model_b_files = [
    "MLP_ION_INDEL1.csv", "MLP_ION_INDEL2.csv", "MLP_ION_INDEL3.csv", "MLP_ION_INDEL4.csv", "MLP_ION_INDEL5.csv"
]

model_labels = ["SHINE-NGS", "GARFIELD-NGS"]  # 两个模型的标签
colors = ['#1f77b4',  # 蓝色（原）
    '#ff7f0e']  # 两条主线的颜色


def process_model(files):
    # 存储所有曲线的fpr和tpr
    all_fpr = []
    all_tpr = []

    for file in files:
        # 读取数据并计算ROC曲线
        df = pd.read_csv(file)
        y_true = df['True_Label']  # 真实标签列名（根据实际调整）
        y_prob = df['Positive_Probability']  # 正类概率列名（根据实际调整）
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

    # 创建统一的FPR网格（0到1，间隔0.001，确保所有曲线对齐）
    fpr_grid = np.linspace(0, 1, 1000)

    # 对每条曲线的TPR在统一网格上插值
    interp_tprs = []
    for fpr, tpr in zip(all_fpr, all_tpr):
        interp_tpr = np.interp(fpr_grid, fpr, tpr)  # 线性插值
        interp_tprs.append(interp_tpr)

    # 计算主线（平均TPR）和上下限（最大/最小TPR）
    tpr_mean = np.mean(interp_tprs, axis=0)  # 平均值作为主线
    tpr_max = np.max(interp_tprs, axis=0)  # 上限
    tpr_min = np.min(interp_tprs, axis=0)  # 下限

    # 计算主线的AUC
    auc_score = auc(fpr_grid, tpr_mean)

    return fpr_grid, tpr_mean, tpr_max, tpr_min, auc_score

plt.figure(figsize=(8, 6))


# 处理模型A并绘图
fpr_a, tpr_a_mean, tpr_a_max, tpr_a_min, auc_a = process_model(model_a_files)
plt.plot(fpr_a, tpr_a_mean, color=colors[0],
         label=f'{model_labels[0]} (AUC={auc_a:.3f})')
plt.fill_between(fpr_a, tpr_a_min, tpr_a_max, color=colors[0], alpha=0.2)  # 阴影区域

# 处理模型B并绘图
fpr_b, tpr_b_mean, tpr_b_max, tpr_b_min, auc_b = process_model(model_b_files)
plt.plot(fpr_b, tpr_b_mean, color=colors[1],
         label=f'{model_labels[1]} (AUC={auc_b:.3f})')
plt.fill_between(fpr_b, tpr_b_min, tpr_b_max, color=colors[1], alpha=0.2)  # 阴影区域

# 绘制随机猜测的对角线
plt.plot([0, 1], [0, 1], 'k--')

# 设置图表细节
plt.title('INDEL (ION)', fontsize=20, fontweight='bold', pad=15)  # 替换为你的标题
plt.legend(loc='lower right', fontsize=15)
plt.tick_params(axis='both', labelsize=20)
plt.show()


# # 2. 处理模型曲线的核心函数
# def process_model(files):
#     all_fpr = []
#     all_tpr = []
#
#     for file in files:
#         # 读取数据（确保CSV列名与实际一致，否则需修改）
#         df = pd.read_csv(file)
#         y_true = df['True_Label']  # 真实标签列名
#         y_prob = df['Positive_Probability']  # 正类概率列名
#         fpr, tpr, _ = roc_curve(y_true, y_prob)
#         all_fpr.append(fpr)
#         all_tpr.append(tpr)
#
#     # 统一FPR网格
#     fpr_grid = np.linspace(0, 1, 1000)
#
#     # 插值对齐曲线
#     interp_tprs = [np.interp(fpr_grid, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]
#
#     # 计算均值、上下限和AUC
#     tpr_mean = np.mean(interp_tprs, axis=0)
#     tpr_max = np.max(interp_tprs, axis=0)
#     tpr_min = np.min(interp_tprs, axis=0)
#     auc_score = auc(fpr_grid, tpr_mean)
#
#     return fpr_grid, tpr_mean, tpr_max, tpr_min, auc_score
#
#
# # 3. 绘图（无网格线）
# plt.figure(figsize=(8, 6))
#
# # 处理模型A（蓝线：SHINE-NGS）
# fpr_a, tpr_a_mean, tpr_a_max, tpr_a_min, auc_a = process_model(model_a_files)
# plt.plot(fpr_a, tpr_a_mean, color=colors[0], linewidth=2,
#          label=f'{model_labels[0]} (AUC={auc_a:.3f})')
#
# # 标记蓝线起始点
# min_fpr_idx = np.argmin(fpr_a)
# min_fpr = fpr_a[min_fpr_idx]
# min_tpr = tpr_a_mean[min_fpr_idx]
# plt.scatter(min_fpr, min_tpr, color=colors[0], s=80, zorder=5, edgecolors='black')
#
# # 添加起始点辅助线
# plt.plot([0, min_fpr], [0, min_tpr], color=colors[0], linestyle='--', alpha=0.6)
#
# # 蓝线阴影区域
# plt.fill_between(fpr_a, tpr_a_min, tpr_a_max, color=colors[0], alpha=0.2)
#
# # 处理模型B（红线：GARFIELD-NGS）
# fpr_b, tpr_b_mean, tpr_b_max, tpr_b_min, auc_b = process_model(model_b_files)
# plt.plot(fpr_b, tpr_b_mean, color=colors[1], linewidth=2,
#          label=f'{model_labels[1]} (AUC={auc_b:.3f})')
# plt.fill_between(fpr_b, tpr_b_min, tpr_b_max, color=colors[1], alpha=0.2)
#
# # 随机猜测线与图表设置（无网格线）
# plt.plot([0, 1], [0, 1], 'k--')
# plt.title('INDEL (ILM)')
# plt.legend(loc='lower right')
#
# # 已删除网格线代码（plt.grid(...)）
#
# plt.show()