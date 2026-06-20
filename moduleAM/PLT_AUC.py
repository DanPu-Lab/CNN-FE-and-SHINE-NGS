import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from pathlib import Path

# ---------------------- 配置参数 ----------------------
# 8个模型CSV文件路径
csv_files = [
    "ResNet34_ION_INDEL2.csv",
    "AlexNet_ION_INDEL.csv",
    "DenseNet121_ION_INDEL.csv",
    "MobileNetV2_ION_INDEL.csv",
    "NASNet_ION_INDEL.csv",
    "RegNet_ION_INDEL.csv",
    "SqueezeNet_ION_INDEL.csv",
    "Xception_ION_INDEL.csv"
]

# csv_files = [
#     "ResNet34_ILM_INDEL2.csv",
#     "AlexNet_ILM_INDEL.csv",
#     "DenseNet121_ILM_INDEL.csv",
#     "MobileNetV2_ILM_INDEL.csv",
#     "NASNet_ILM_INDEL.csv",
#     "RegNet_ILM_INDEL.csv",
#     "SqueezeNet_ILM_INDEL.csv",
#     "Xception_ILM_INDEL.csv"
# ]


# 模型名简写规则
def simplify_model_name(file_name):
    name_parts = file_name.split('_')
    model_name = name_parts[0]
    if 'DenseNet' in model_name:
        return 'DenseNet'
    elif 'MobileNet' in model_name:
        return 'MobileNet'
    else:
        return model_name


# 曲线颜色（8种不同颜色，统一实线）
colors = [
    '#1f77b4',  # 蓝色（原）
    '#ff7f0e',  # 橙色（原）
    '#2ca02c',  # 绿色（原）
    '#d62728',  # 红色（原）
    '#9467bd',  # 紫色（原）
    '#8c564b',  # 棕色（原）
    '#e377c2',  # 粉色（原）
    '#ff6b6b',  # 珊瑚红（原）
]


# ---------------------- 计算ROC曲线和AUC ----------------------
def calculate_roc_auc(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"警告：文件 {csv_file} 未找到，已跳过")
        return None, None, None

    required_cols = ['True_Label', 'Positive_Probability']
    if not set(required_cols).issubset(df.columns):
        print(f"警告：文件 {csv_file} 缺少 {required_cols}，已跳过")
        return None, None, None

    y_true = df['True_Label'].astype(int)
    y_prob = df['Positive_Probability'].clip(0, 1)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


# ---------------------- 绘制AUC曲线（最终样式） ----------------------
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2

fig, ax = plt.subplots(figsize=(8, 6))

# 遍历每个模型绘制实线ROC曲线
for i, file in enumerate(csv_files):
    fpr, tpr, roc_auc = calculate_roc_auc(file)
    if fpr is None:
        continue

    model_name = simplify_model_name(Path(file).stem)

    ax.plot(
        fpr, tpr,
        color=colors[i],
        linestyle='-',  # 统一实线
        linewidth=2.5,
        label=f'{model_name} (AUC = {roc_auc:.3f})'
    )

# 图表配置（去掉随机预测基准线）
ax.set_title('ILM (INDEL)', fontsize=20, fontweight='bold', pad=15)
ax.set_xlabel('FPR', fontsize=20)
ax.set_ylabel('TPR', fontsize=20)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xticks(np.arange(0, 1.1, 0.2))
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.tick_params(axis='both', labelsize=20)

# 样式调整（四面黑色边框，无网格线）
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.grid(False)

# 图例移至图内右下角，添加半透明背景避免遮挡曲线
ax.legend(
    loc='lower right',  # 右下角位置
    fontsize=15,
    framealpha=0.9,  # 图例背景透明度
    facecolor='white',  # 图例背景色
    edgecolor='gray',  # 图例边框色
    labelspacing=0.5  # 图例项间距
)

plt.tight_layout()
plt.savefig('models_auc_roc_final.png', dpi=400, bbox_inches='tight')
plt.close()

print("最终版AUC曲线图已生成：models_auc_roc_final.png")