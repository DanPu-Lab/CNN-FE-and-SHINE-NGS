
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ---------------------- 1. 核心配置（复用你的文件路径和类别） ----------------------
categories = ["ILM INDEL", "ILM SNP", "ION INDEL", "ION SNP"]
methods = ["SHINE-NGS", "GARFIELD-NGS"]
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

# 配色（保持你之前的设置：红色、灰色）
colors = ['#1f77b4',  # 蓝色（原）
    '#ff7f0e' ]  # SHINE-NGS=红色，GARRIELD-NGS=灰色
OFO_DEFINITION = 1  # 固定为NGS假阳性核心指标：FP/(TP+FP)（过度调用率）

# ---------------------- 2. 计算OFO（假阳性过度调用率） ----------------------
ofo_data = []
for cat in categories:
    for method_idx, (method, files) in enumerate(file_config[cat].items()):
        ofo_scores = []
        for file in files:
            try:
                df = pd.read_csv(file)
                y_true = df["True_Label"]  # 1=真实变异，0=非变异
                y_pred = df["Final_Prediction"]  # 1=预测变异，0=预测非变异

                # 计算混淆矩阵（TN, FP, FN, TP）
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

                # 计算OFO（避免分母为0）
                if (tp + fp) == 0:
                    ofo = 0.0  # 无预测为变异的样本，OFO为0
                else:
                    ofo = fp / (tp + fp)  # 假阳性占所有预测变异的比例（核心定义）

                ofo_scores.append(ofo)
            except Exception as e:
                print(f"⚠️  处理文件 {file} 时出错：{str(e)}")
                ofo_scores.append(0.0)  # 出错时填充0，不影响整体图表

        # 计算该方法在当前类别的OFO均值（单个文件则直接取该值）
        avg_ofo = sum(ofo_scores) / len(ofo_scores)
        ofo_data.append({
            "Category": cat,
            "Method": method,
            "OFO": avg_ofo
        })

# 转换为DataFrame，方便绘图
df_ofo = pd.DataFrame(ofo_data)

# ---------------------- 3. 绘制OFO柱状图（保持原风格） ----------------------
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = range(len(categories))  # 4个类别的x轴基础位置

# 分组绘制柱状图（每个类别2个细柱）
for method_idx, method in enumerate(methods):
    ofo_values = df_ofo[df_ofo["Method"] == method]["OFO"].values
    ax.bar(
        [i + method_idx * bar_width for i in index],  # 细柱横向偏移，避免重叠
        ofo_values,
        width=bar_width,
        color=colors[method_idx],
        label=method,
        alpha=0.8  # 轻微透明度，更美观
    )

# ---------------------- 4. 图表美化（适配OFO指标） ----------------------
# Y轴标签（明确OFO含义，符合科研图表规范）
# ax.set_ylabel("OFO (False Positive Overcall Rate)", fontsize=12, fontweight="bold")
# # 标题（说明图表主题）
# ax.set_title("OFO Comparison Across Platform and Variant Types", fontsize=14, fontweight="bold", pad=20)
plt.rcParams['font.sans-serif'] = ['Arial']
# X轴刻度和标签（居中对齐两个细柱）
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(categories, fontsize=20)

# Y轴范围（OFO取值0~1，设置0~0.5更聚焦差异，可根据实际数据调整）
ax.set_yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
ax.set_yticklabels([0.00, 0.02, 0.04, 0.06, 0.08, 0.10], fontsize=20) # 若你的OFO值较大，可改为0~1

# 网格线（横向，增强可读性）
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)

# 自定义图例（右上方带颜色矩形，与原风格一致）
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.8) for i in range(len(methods))]
ax.legend(handles, methods, loc="upper right", fontsize=15, frameon=False)

# 调整布局，避免标签截断
plt.tight_layout()

# 保存高清图片（300dpi，适合论文/报告）
plt.savefig("OFO_False_Positive_Comparison.png", dpi=400, bbox_inches="tight")
print("💾 OFO假阳性对比图已保存：OFO_False_Positive_Comparison.png")

# 打印OFO数值汇总（方便核对）
print("\n" + "=" * 60)
print("📊 各「类别+方法」的OFO值（假阳性过度调用率）")
print("=" * 60)
print(df_ofo.round(4).to_string(index=False))

plt.show()