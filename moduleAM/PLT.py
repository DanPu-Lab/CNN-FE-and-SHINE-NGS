import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 读取两个CSV文件
df1 = pd.read_csv('Resnet34_raw_ILM_INDEL.csv')
df2 = pd.read_csv('Resnet34_ILM_INDEL2.csv')

# 提取真实标签和正类概率
y1 = df1['True_Label']
y_score1 = df1['Positive_Probability']
y2 = df2['True_Label']
y_score2 = df2['Positive_Probability']

# 计算ROC曲线的FPR、TPR和AUC
fpr1, tpr1, _ = roc_curve(y1, y_score1)
auc1 = roc_auc_score(y1, y_score1)
fpr2, tpr2, _ = roc_curve(y2, y_score2)
auc2 = roc_auc_score(y2, y_score2)

# 设置中文字体（避免乱码）
plt.rcParams["font.family"] = ['Arial']
plt.figure(figsize=(8, 6))

# 绘制ROC曲线
plt.plot(fpr2, tpr2, color='#1f77b4', lw=2, label=f'SHINE-NGS (AUC = {auc2:.4f})')
plt.plot(fpr1, tpr1, color='#ff7f0e', lw=2, label=f'No FE (AUC = {auc1:.4f})')
# 绘制随机猜测基准线
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

# 设置坐标轴和标题
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR', fontsize=20)
plt.ylabel('TPR', fontsize=20)
plt.title('INDEL', fontsize=20, fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=15)
plt.tick_params(axis='both', labelsize=20)

# 保存并显示图像
plt.savefig('auc_comparison.png', dpi=400, bbox_inches='tight')
plt.show()