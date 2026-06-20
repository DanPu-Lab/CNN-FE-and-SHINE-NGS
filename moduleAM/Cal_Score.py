import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score, accuracy_score,
    precision_recall_curve, auc, confusion_matrix
)

# ---------------------- 1. 核心配置（已按你的要求固定） ----------------------
# categories = ["ILM INDEL", "ILM SNP", "ION INDEL", "ION SNP"]
#
# file_config = {
#     "ILM INDEL": [
#         ["Resnet34_ILM_INDEL1.csv", "Resnet34_ILM_INDEL2.csv", "Resnet34_ILM_INDEL3.csv", "Resnet34_ILM_INDEL4.csv",
#          "Resnet34_ILM_INDEL5.csv"],
#         ["MLP_ILM_INDEL1.csv", "MLP_ILM_INDEL2.csv", "MLP_ILM_INDEL3.csv", "MLP_ILM_INDEL4.csv", "MLP_ILM_INDEL5.csv"]
#     ],
#     "ILM SNP": [
#         ["Resnet34_ILM_SNP1.csv", "Resnet34_ILM_SNP2.csv", "Resnet34_ILM_SNP3.csv", "Resnet34_ILM_SNP4.csv",
#          "Resnet34_ILM_SNP5.csv"],
#         ["MLP_ILM_SNP1.csv", "MLP_ILM_SNP2.csv", "MLP_ILM_SNP3.csv", "MLP_ILM_SNP4.csv", "MLP_ILM_SNP5.csv"]
#     ],
#     "ION INDEL": [
#         ["Resnet34_ION_INDEL1.csv", "Resnet34_ION_INDEL2.csv", "Resnet34_ION_INDEL3.csv", "Resnet34_ION_INDEL4.csv",
#          "Resnet34_ION_INDEL5.csv"],
#         ["MLP_ION_INDEL1.csv", "MLP_ION_INDEL2.csv", "MLP_ION_INDEL3.csv", "MLP_ION_INDEL4.csv", "MLP_ION_INDEL5.csv"]
#     ],
#     "ION SNP": [
#         ["Resnet34_ION_SNP1.csv", "Resnet34_ION_SNP2.csv", "Resnet34_ION_SNP3.csv", "Resnet34_ION_SNP4.csv",
#          "Resnet34_ION_SNP5.csv"],
#         ["MLP_ION_SNP1.csv", "MLP_ION_SNP2.csv", "MLP_ION_SNP3.csv", "MLP_ION_SNP4.csv", "MLP_ION_SNP5.csv"]
#     ]
# }

categories = ["ILM", "ION"]

file_config = {
    "ILM": [
        ["Resnet34_ILM_INDEL2.csv"],
        ["Resnet34_raw_ILM_INDEL.csv"]
    ],
    "ION": [
        ["Resnet34_ION_INDEL2.csv"],
        ["Resnet34_raw_ION_INDEL.csv"]
    ]
}

model_names = ["Resnet34", "raw"]
true_label_col = "True_Label"
pred_label_col = "Final_Prediction"
prob_col = "Positive_Probability"
OFO_DEFINITION = 1  # 1=FP/(TP+FP)（推荐），2=FP/(TN+FP)


# ---------------------- 2. 指标计算函数（无修改） ----------------------
def calculate_all_metrics(df):
    y_true = df[true_label_col]
    y_pred = df[pred_label_col]
    y_prob = df[prob_col]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # AUC（避免单类别报错）
    auc_score = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    acc = accuracy_score(y_true, y_pred)

    # AUPRG
    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    auprg = auc(recall, precision)

    # OFO
    if OFO_DEFINITION == 1:
        ofo = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    else:
        ofo = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    # F1-minor（自动识别少数类）
    class_counts = y_true.value_counts()
    minor_class = class_counts.idxmin() if len(class_counts) > 1 else 0
    f1_minor = f1_score(y_true, y_pred, pos_label=minor_class)

    # BACC
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc = (sensitivity + specificity) / 2

    # NPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "AUC": auc_score, "MCC": mcc, "F1": f1, "ACC": acc, "AUPRG": auprg,
        "OFO": ofo, "F1-minor": f1_minor, "BACCOFO": ofo, "F1-minor": f1_minor, "BACC": bacc, "NPV": npv,
        "TN": tn, "FN": fn
    }


# ---------------------- 3. 批量处理40个CSV文件（无修改） ----------------------
all_results = []

for cat in categories:
    model_a_files, model_b_files = file_config[cat]
    for model_idx, (model_name, files) in enumerate(zip(model_names, [model_a_files, model_b_files])):
        for file in files:
            try:
                df = pd.read_csv(file)
                required_cols = [true_label_col, pred_label_col, prob_col]
                missing_cols = set(required_cols) - set(df.columns)
                if missing_cols:
                    raise ValueError(f"缺少列：{', '.join(missing_cols)}")

                metrics = calculate_all_metrics(df)
                all_results.append({
                    "Category": cat,
                    "Model": model_name,
                    "Filename": file,
                    **metrics
                })
            except Exception as e:
                print(f"⚠️  处理文件 {file} 时出错：{str(e)}")

results_df = pd.DataFrame(all_results)

# ---------------------- 4. 修复后的结果汇总（兼容所有pandas版本） ----------------------
# 分两步聚合：先处理连续型指标（均值+标准差），再处理计数型指标（总和）
continuous_metrics = ["AUC", "MCC", "F1", "ACC", "AUPRG", "OFO", "F1-minor", "BACC", "NPV"]
count_metrics = ["TN", "FN"]

# 步骤1：连续型指标（均值+标准差）
summary_continuous = results_df.groupby(["Category", "Model"])[continuous_metrics].agg(["mean", "std"]).round(4)

# 步骤2：计数型指标（总和）
summary_count = results_df.groupby(["Category", "Model"])[count_metrics].agg("sum").round(4)

# 步骤3：合并两个结果（按索引对齐）
summary_df = pd.concat([summary_continuous, summary_count], axis=1)

# ---------------------- 5. 结果输出（无修改） ----------------------
print("=" * 80)
print("📊 按类别+模型汇总结果（均值±标准差 / 总和）")
print("=" * 80)
print(summary_df)

# 保存详细结果
results_df.to_csv("NGS_Variant_Detection_Detailed_Metrics.csv", index=False, encoding="utf-8")
print("\n💾 详细结果已保存：NGS_Variant_Detection_Detailed_Metrics.csv")

# 保存汇总结果（ flatten列名，更易读）
summary_df.columns = [f"{col}_{agg}" for col, agg in summary_df.columns if col in continuous_metrics] + count_metrics
summary_df.to_csv("NGS_Variant_Detection_Summary_Metrics.csv", encoding="utf-8")
print("💾 汇总结果已保存：NGS_Variant_Detection_Summary_Metrics.csv")

# # 可选：模型整体性能对比
# overall_continuous = results_df.groupby("Model")[continuous_metrics].agg(["mean", "std"]).round(4)
# overall_count = results_df.groupby("Model")[count_metrics].agg(4)
# overall_count = results_df.groupby("Model")[count_metrics].agg("sum").round(4)
# overall_summary = pd.concat([overall_continuous, overall_count], axis=1)
#
# print("\n" + "=" * 80)
# print("📈 两个模型整体性能对比（所有类别均值±标准差 / 总和）")
# print("=" * 80)
# print(overall_summary)