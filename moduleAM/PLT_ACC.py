import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---------------------- 1. 定义8个CSV文件路径 ----------------------
file_paths = [
    "Resnet34_ILM_INDEL2.csv",
    "MLP_ILM_INDEL5.csv",
    "Resnet34_ILM_SNP2.csv",
    "MLP_ILM_SNP3.csv",
    "Resnet34_ION_INDEL2.csv",
    "MLP_ION_INDEL3.csv",
    "Resnet34_ION_SNP5.csv",
    "MLP_ION_SNP2.csv"
]

# ---------------------- 2. 定义颜色（与示例风格匹配，最后一个点为红色） ----------------------
colors = ["blue", "lightblue", "gray", "orange", "yellow", "green", "teal", "red"]

# ---------------------- 3. 计算每个文件的Accuracy并绘图 ----------------------
plt.figure(figsize=(4, 8))  # 调整画布比例，匹配示例风格

for i, file in enumerate(file_paths):
    # 读取CSV并提取真实标签和预测标签
    df = pd.read_csv(file)
    y_true = df["True_Label"]
    y_pred = df["Final_Prediction"]

    # 计算Accuracy
    acc = accuracy_score(y_true, y_pred)

    # 绘制散点（纵轴用索引表示不同文件，横轴为Accuracy）
    plt.scatter(
        acc,  # 横轴：Accuracy值
        i,  # 纵轴：文件索引（实现垂直分布）
        color=colors[i],
        s=50,  # 点的大小
        zorder=3  # 确保点在网格上方
    )

# ---------------------- 4. 图表美化 ----------------------
plt.xlim(0.8, 1)  # 横轴范围0-1
plt.ylim(-0.5, 7.5)  # 纵轴范围适配8个点（索引0-7）
plt.yticks([])  # 隐藏纵轴刻度
plt.xlabel("Accuracy")
plt.title("Accuracy")
plt.grid(axis="x", linestyle="--", alpha=0.5, zorder=1)  # 仅显示横轴网格
plt.tight_layout()

# 保存高清图片
plt.savefig("accuracy_scatter.png", dpi=300, bbox_inches="tight")
plt.show()