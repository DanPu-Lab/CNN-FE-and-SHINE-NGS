import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from matplotlib.colors import rgb2hex, hex2color

# # ---------------------- 核心配置：10个原始特征（无黑色配色） ----------------------
# original_features = [
#     'BaseQRankSum', 'ReadPosRankSum', 'DP', 'FS', 'MQ',
#     'MQRankSum', 'QD', 'SOR', 'QUAL', 'GQ'
# ]
original_features = [
    'FDP', 'FAO', 'QD', 'FSAF', 'FSAR', 'FXX', 'LEN', 'HRUN', 'RBI',
    'VARB', 'STB', 'STBP', 'MLLD', 'SSEN', 'SSEP', 'SSSB', 'QUAL', 'GQ'
]
# 10种鲜明配色（无黑色/深灰色，辨识度高）
# original_colors = [
#     '#1f77b4',  # 蓝色
#     '#ff7f0e',  # 橙色
#     '#2ca02c',  # 绿色
#     '#d62728',  # 红色
#     '#9467bd',  # 紫色
#     '#8c564b',  # 棕色（非黑色）
#     '#e377c2',  # 粉色
#     '#ff6b6b',  # 珊瑚红（替换原深灰色#7f7f7f）
#     '#bcbd22',  # 黄绿色
#     '#17becf'   # 青色
# ]
original_colors = [
    '#1f77b4',  # 蓝色（原）
    '#ff7f0e',  # 橙色（原）
    '#2ca02c',  # 绿色（原）
    '#d62728',  # 红色（原）
    '#9467bd',  # 紫色（原）
    '#8c564b',  # 棕色（原）
    '#e377c2',  # 粉色（原）
    '#ff6b6b',  # 珊瑚红（原）
    '#bcbd22',  # 黄绿色（原）
    '#17becf',  # 青色（原）
    '#aec7e8',  # 浅蓝（新增）
    '#ffbb78',  # 浅橙（新增）
    '#98df8a',  # 浅绿（新增）
    '#ff9896',  # 浅红（新增）
    '#c5b0d5',  # 浅紫（新增）
    '#c49c94',  # 浅棕（新增）
    '#f7b6d3',  # 浅粉（新增）
    '#9edae5'   # 浅青（新增）
]  # 18个颜色，与原始特征一一对应
orig_feat_color = dict(zip(original_features, original_colors))

# ---------------------- 辅助函数（不变） ----------------------
def match_original_features(new_feat_name):
    matched_orig_feats = []
    for orig_feat in original_features:
        if orig_feat in new_feat_name:
            matched_orig_feats.append(orig_feat)
    return matched_orig_feats

def blend_colors(color_hex_list):
    if len(color_hex_list) == 1:
        return color_hex_list[0]
    rgb_list = [np.array(hex2color(color)) for color in color_hex_list]
    blended_rgb = np.mean(rgb_list, axis=0)
    return rgb2hex(blended_rgb)

# ---------------------- 特征重要性计算（不变） ----------------------
input_file = '../finalFE/FE_ION_INDEL_Train.csv'
df = pd.read_csv(input_file)
X = df.iloc[:, 1:].values
y = df['Class'].apply(lambda x: 1 if x == 'T' else 0).values

lgb_train = lgb.Dataset(X, y)
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}
lgbm_model = lgb.train(lgb_params, lgb_train, num_boost_round=100)
lgb_importance = lgbm_model.feature_importance()
lgb_feature_names = df.columns[1:]
lgb_importance = lgb_importance / np.max(lgb_importance)

# 筛选Top20特征
lgb_important_features = [(name, imp) for name, imp in zip(lgb_feature_names, lgb_importance) if imp > 0.001]
lgb_important_features_sorted = sorted(lgb_important_features, key=lambda x: x[1], reverse=True)
top_n = 20
lgb_top_features = lgb_important_features_sorted[:top_n]
features = [f[0] for f in lgb_top_features]
importances = [f[1] for f in lgb_top_features]

# 特征重要度
print("="*60)
print("LightGBM Top20 特征名称及归一化重要度")
print("="*60)
print(f"{'排名':<6}{'特征名称':<30}{'归一化重要度':<15}")
print("-"*60)
for idx, (feat, imp) in enumerate(zip(features, importances), start=1):
    print(f"{idx:<6}{feat:<30}{imp:.4f}")  # 保留4位小数，便于对比
print("="*60)

# # ---------------------- 特征颜色分配（不变） ----------------------
# feature_colors = []
# match_records = []
# for feat in features:
#     if feat in original_features:
#         color = orig_feat_color[feat]  # 无黑色的原始特征颜色
#         match_records.append((feat, '原始特征', [feat]))
#     else:
#         matched_orig = match_original_features(feat)
#         if matched_orig:
#             matched_colors = [orig_feat_color[orig_feat] for orig_feat in matched_orig]
#             color = blend_colors(matched_colors)
#             match_records.append((feat, '新增特征', matched_orig))
#         else:
#             color = '#cccccc'  # 未匹配新增特征：浅灰色（非黑色）
#             match_records.append((feat, '新增特征', '无匹配'))
#     feature_colors.append(color)

# ---------------------- 绘制柱状图（不变） ----------------------
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2

fig, ax = plt.subplots(figsize=(16, 12))

bars = ax.bar(
    features, importances,
    color=original_colors,
    edgecolor='black',  # 柱子边框仍用黑色（不影响原始特征颜色）
    linewidth=0.8,
    alpha=0.9
)

ax.set_ylabel('Important', fontsize=20, fontweight='bold')
ax.set_xlabel('Feature', fontsize=20, fontweight='bold')
ax.set_ylim(0, max(importances) * 1.15)

plt.xticks(rotation=45, ha='right', fontsize=20)
plt.yticks(fontsize=20)

# 四面黑色边框（图表框架，不影响特征颜色）
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
# 1. 消除X轴左右边距（最关键的一步）
ax.margins(x=0)

# 2. 手动设置X轴左边界，让第一个柱子紧贴Y轴
ax.set_xlim(left=-0.5)

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('top20_feature_importance_no_black.png', dpi=400, bbox_inches='tight')
plt.close()

# # ---------------------- 输出日志 ----------------------
# print("特征-颜色匹配日志（原始特征无黑色）：")
# print("="*80)
# for feat, feat_type, matched in match_records:
#     print(f"特征：{feat:25} 类型：{feat_type:6} 匹配原始特征：{str(matched):25}")
# print("\n✅ 无黑色原始特征的图表已生成：top20_feature_importance_no_black.png")