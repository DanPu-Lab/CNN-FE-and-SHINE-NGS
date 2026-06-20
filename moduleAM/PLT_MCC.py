import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef

# ---------------------- 1. 定义文件与类别映射 ----------------------
categories = ["ILM INDEL", "ILM SNP", "ION INDEL", "ION SNP"]
methods = ["SHINE-NGS", "GARFIELD-NGS"]
# 替换为实际文件路径，格式：{类别: {方法: [文件1, 文件2, ...]}}
file_config = {
    "ILM INDEL": {
        "SHINE-NGS": ["Resnet34_ILM_INDEL2.csv"],
        "GARFIELD-NGS": ["MLP_ILM_INDEL5.csv"]
    },
    "ILM SNP": {
        "SHINE-NGS": ["Resnet34_ILM_SNP2.csv"],
        "GARFIELD-NGS": ["MLP_ILM_SNP3.csv"]
    },
    "ION INDEL": {
        "SHINE-NGS": ["Resnet34_ION_INDEL2.csv"],
        "GARFIELD-NGS": ["MLP_ION_INDEL3.csv"]
    },
    "ION SNP": {
        "SHINE-NGS": ["Resnet34_ION_SNP5.csv"],
        "GARFIELD-NGS": ["MLP_ION_SNP2.csv"]
    }
}
# categories = ["ILM","ION"]
# methods = ["SHINE-NGS", "No FE"]
# file_config = {
#     "ILM": {
#         "SHINE-NGS": ["Resnet34_ILM_INDEL2.csv"],
#         "No FE": ["Resnet34_raw_ILM_INDEL.csv"]
#     },
#     "ION": {
#         "SHINE-NGS": ["Resnet34_ION_INDEL2.csv"],
#         "No FE": ["Resnet34_raw_ION_INDEL.csv"]
#     }
# }

# 配色（匹配示例：红色、灰色）
colors = ['#1f77b4',  # 蓝色（原）
    '#ff7f0e' ]  # 红色、灰色

# ---------------------- 2. 计算每个文件的MCC分数 ----------------------
mcc_data = []
for cat in categories:
    for method_idx, (method, files) in enumerate(file_config[cat].items()):
        mcc_scores = []
        for file in files:
            df = pd.read_csv(file)
            y_true = df["True_Label"]
            y_pred = df["Final_Prediction"]
            mcc = matthews_corrcoef(y_true, y_pred)
            mcc_scores.append(mcc)
        # 取平均值（若多个文件代表重复实验）
        avg_mcc = sum(mcc_scores) / len(mcc_scores)
        mcc_data.append({
            "Category": cat,
            "Method": method,
            "MCC": avg_mcc
        })

# 转换为DataFrame
df_mcc = pd.DataFrame(mcc_data)

# ---------------------- 3. 绘制柱状图 ----------------------
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = range(len(categories))

# 分组绘制
for method_idx, method in enumerate(methods):
    mcc_values = df_mcc[df_mcc["Method"] == method]["MCC"].values
    ax.bar(
        [i + method_idx * bar_width for i in index],
        mcc_values,
        width=bar_width,
        color=colors[method_idx],
        label=method
    )
plt.rcParams['font.sans-serif'] = ['Arial']
# 美化图表
# ax.set_ylabel("MCC", fontsize=12, fontweight="bold")
# ax.set_title("MCC Comparison Across Platform and Models", fontsize=14, fontweight="bold")
ax.set_xticks([i + bar_width/2 for i in index])
ax.set_xticklabels(categories, fontsize=20)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
# 自定义图例（右上方带颜色矩形）
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(methods))]
ax.legend(handles, methods, loc="upper right", fontsize=15, frameon=False)

plt.tight_layout()
plt.savefig("MCC_Comparison.png", dpi=400, bbox_inches="tight")
plt.show()