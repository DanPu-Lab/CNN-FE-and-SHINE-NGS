import pandas as pd
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score, accuracy_score,
    balanced_accuracy_score, confusion_matrix, average_precision_score
)
import numpy as np

# -------------------------- 1. 配置参数（核心：选择OFO计算方式） --------------------------
# OFO_DEFINITION=1 → OFO = fp/(tp+fp)（预测为阳性中的假阳性比例）
# OFO_DEFINITION≠1 → OFO = fp/(tn+fp)（所有阴性样本中的假阳性比例，即假阳性率FPR）
OFO_DEFINITION = 1  # 可根据需求修改为1或其他值

# -------------------------- 2. 定义所有CSV文件路径（16个文件） --------------------------
csv_files = [
    # ILM系列
    "ResNet34_ILM_INDEL2.csv",
    "AlexNet_ILM_INDEL.csv",
    "DenseNet121_ILM_INDEL.csv",
    "MobileNetV2_ILM_INDEL.csv",
    "NASNet_ILM_INDEL.csv",
    "RegNet_ILM_INDEL.csv",
    "SqueezeNet_ILM_INDEL.csv",
    "Xception_ILM_INDEL.csv",
    # ION系列
    "ResNet34_ION_INDEL2.csv",
    "AlexNet_ION_INDEL.csv",
    "DenseNet121_ION_INDEL.csv",
    "MobileNetV2_ION_INDEL.csv",
    "NASNet_ION_INDEL.csv",
    "RegNet_ION_INDEL.csv",
    "SqueezeNet_ION_INDEL.csv",
    "Xception_ION_INDEL.csv"
]


# -------------------------- 3. 定义指标计算函数（适配OFO新逻辑） --------------------------
def calculate_metrics(df, threshold=0.5):
    """
    计算单个CSV文件的所有指标（含TN、FN，OFO按二选一逻辑计算）
    适配列名：True_Label（真实标签）、Positive_Probability（正例预测概率）
    """
    # 提取核心列并删除缺失值
    df = df[["True_Label", "Positive_Probability"]].dropna()
    y_true = df["True_Label"].astype(int)  # 真实标签（0/1）
    y_score = df["Positive_Probability"].astype(float)  # 正例预测概率（0~1）
    y_pred = (y_score >= threshold).astype(int)  # 预测标签（基于阈值）

    # 混淆矩阵核心值
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 按定义计算OFO（避免除零错误）
    if OFO_DEFINITION == 1:
        ofo = fp / (tp + fp) if (tp + fp) > 0 else 0.0  # 分母：所有预测为阳性的样本数
    else:
        ofo = fp / (tn + fp) if (tn + fp) > 0 else 0.0  # 分母：所有真实为阴性的样本数（假阳性率FPR）

    # 所有指标（OFO按新逻辑计算）
    metrics = {
        "AUC": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ACC": accuracy_score(y_true, y_pred),
        "AUPRG": average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        "TN": tn,  # 真阴性数
        "FN": fn,  # 假阴性数
        "OFO": round(ofo, 4),  # 按定义计算，保留4位小数
        "F1-minor": f1_score(y_true, y_pred,
                             pos_label=1 if (y_true == 1).sum() < (y_true == 0).sum() else 0),
        "BACC": balanced_accuracy_score(y_true, y_pred),
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else np.nan  # 阴性预测值
    }
    return metrics


# -------------------------- 4. 批量计算所有文件指标 --------------------------
results = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        # 可选：改用CSV中的F1最优阈值（取消注释即可）
        # optimal_threshold = df["F1_Optimal_Threshold"].iloc[0]
        # metrics = calculate_metrics(df, threshold=optimal_threshold)

        # 默认用0.5阈值计算
        metrics = calculate_metrics(df)
        metrics["文件名"] = file
        results.append(metrics)
        print(f"✅ 成功处理：{file}")
    except Exception as e:
        print(f"❌ 处理失败 {file}：{str(e)}")

# -------------------------- 5. 整理结果并输出 --------------------------
result_df = pd.DataFrame(results)[
    ["文件名", "AUC", "MCC", "F1", "ACC", "AUPRG", "BACC", "NPV",
     "TN", "FN", "OFO", "F1-minor"]
].round(4)  # 所有指标保留4位小数

# 保存结果到Excel（文件名标注OFO计算方式）
excel_filename = f"模型指标汇总（OFO定义{OFO_DEFINITION}）.xlsx"
result_df.to_excel(excel_filename, index=False)

# 打印结果和OFO计算说明
print(f"\n📊 所有文件指标计算完成！OFO计算方式：")
print(f"   OFO_DEFINITION={OFO_DEFINITION} → OFO = {'fp/(tp+fp)' if OFO_DEFINITION == 1 else 'fp/(tn+fp)'}")
print("\n指标结果：")
print(result_df)