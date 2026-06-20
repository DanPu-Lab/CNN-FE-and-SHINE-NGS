import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
def FE(data):
    df = data.iloc[:, 1:]

    print("开始数学变换")

    math_transforms = {}
    for col in df.select_dtypes(include = [np.number]).columns:
        math_transforms[f'log_{col}'] = np.log1p(df[col].replace(0, 1e-5)) #对数
        math_transforms[f'sqrt_{col}'] = np.sqrt(df[col].clip(lower = 0)) #平方根
        math_transforms[f'exp_{col}'] = np.exp(df[col].clip(upper = 20)) #指数
        math_transforms[f'inverse_{col}'] = 1 / (df[col].replace(0, 1e-5)) #倒数
        math_transforms[f'abs_{col}'] = np.abs(df[col]) #绝对值
        math_transforms[f'{col}_power_2'] = df[col] ** 2  # 平方
        math_transforms[f'{col}_power_3'] = df[col] ** 3  # 立方
        math_transforms[f'sin_{col}'] = np.sin(df[col])  # 正弦
        math_transforms[f'cos_{col}'] = np.cos(df[col])  # 余弦
        math_transforms[f'tan_{col}'] = np.tan(df[col].clip(-np.pi / 2, np.pi / 2))  # 正切
        math_transforms[f'sinh_{col}'] = np.sinh(df[col].clip(upper=20))  # 双曲正弦
        math_transforms[f'cosh_{col}'] = np.cosh(df[col].clip(upper=20))  # 双曲余弦
        math_transforms[f'tanh_{col}'] = np.tanh(df[col])  # 双曲正切
        math_transforms[f'arcsin_{col}'] = np.arcsin(df[col].clip(-1, 1))  # 反正弦
        math_transforms[f'arccos_{col}'] = np.arccos(df[col].clip(-1, 1))  # 反余弦
        math_transforms[f'arctan_{col}'] = np.arctan(df[col])  # 反正切
        math_transforms[f'exp10_{col}'] = 10 ** df[col].clip(upper=20)  # 10 的幂
        math_transforms[f'log10_{col}'] = np.log10(df[col].replace(0, 1e-5))  # 以 10 为底的对数
        math_transforms[f'log2_{col}'] = np.log2(df[col].replace(0, 1e-5))  # 以 2 为底的对数
        math_transforms[f'floor_{col}'] = np.floor(df[col])  # 向下取整
        math_transforms[f'ceil_{col}'] = np.ceil(df[col])  # 向上取整
        math_transforms[f'round_{col}'] = np.round(df[col])  # 四舍五入
        math_transforms[f'abs_inv_{col}'] = 1 / (np.abs(df[col]) + 1e-5)  # 绝对值的倒数

    math_transforms_df = pd.DataFrame(math_transforms)

    print("数学变换完成")

    print('-------------------------------------------------------------------------------------------')

    print("开始分箱操作")

    bin_transforms = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        # 等宽分箱
        bin_transforms[f'bin_{col}'] = pd.cut(df[col], bins=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
        bin_transforms[f'bin_5_{col}'] = pd.cut(df[col], bins=5,
                                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                                                duplicates='drop')
        # 等频分箱
        bin_transforms[f'qbin_{col}'] = pd.qcut(df[col].rank(method='first'), q=3, labels=['Low', 'Medium', 'High'],
                                                duplicates='drop')
        bin_transforms[f'qbin_5_{col}'] = pd.qcut(df[col].rank(method='first'), q=5,
                                                  labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')

        # 基于标准差的分箱（根据偏离均值的标准差区间分箱）
        mean = df[col].mean()
        std = df[col].std()
        bin_transforms[f'std_bin_{col}'] = pd.cut(df[col],
                                                  bins=[-float('inf'), mean - std, mean, mean + std, float('inf')],
                                                  labels=['Below Avg', 'Avg', 'Above Avg', 'High'], duplicates='drop')

    bin_transforms_df = pd.DataFrame(bin_transforms)

    # 独热编码
    bin_transforms_df = pd.get_dummies(bin_transforms_df, prefix=bin_transforms_df.columns, dtype = int)

    print("分箱操作和独热编码完成")

    print('-------------------------------------------------------------------------------------------')

    print("开始特征交互")

    interaction_transforms = {}
    for f1, f2 in combinations(df.select_dtypes(include=[np.number]).columns, 2):
        interaction_transforms[f'{f1}_x_{f2}'] = df[f1] * df[f2]  # 乘积
        interaction_transforms[f'{f1}_plus_{f2}'] = df[f1] + df[f2]  # 加和
        interaction_transforms[f'{f1}_minus_{f2}'] = df[f1] - df[f2]  # 差值
        interaction_transforms[f'{f1}_div_{f2}'] = df[f1] / (df[f2].replace(0, 1e-5))  # 除法，避免除零
        interaction_transforms[f'{f1}_pow2_plus_{f2}_pow2'] = (df[f1] ** 2) + (df[f2] ** 2)  # 平方和
        interaction_transforms[f'{f1}_pow2_minus_{f2}_pow2'] = (df[f1] ** 2) - (df[f2] ** 2)  # 平方差
        interaction_transforms[f'{f1}_x_log_{f2}'] = df[f1] * np.log1p(df[f2].replace(0, 1e-5))  # 乘以对数
        interaction_transforms[f'log_{f1}_x_{f2}'] = np.log1p(df[f1].replace(0, 1e-5)) * df[f2]  # 对数乘以另一个特征
        interaction_transforms[f'{f1}_abs_plus_{f2}'] = np.abs(df[f1]) + df[f2]  # 绝对值加法
        interaction_transforms[f'{f1}_abs_x_{f2}_abs'] = np.abs(df[f1]) * np.abs(df[f2])  # 绝对值乘积
        interaction_transforms[f'{f1}_sqrt_x_{f2}_sqrt'] = np.sqrt(df[f1].clip(lower=0)) * np.sqrt(
            df[f2].clip(lower=0))  # 平方根乘积
        interaction_transforms[f'{f1}_sin_plus_{f2}_cos'] = np.sin(df[f1]) + np.cos(df[f2])  # 正弦和余弦之和
        interaction_transforms[f'{f1}_exp_x_{f2}'] = np.exp(df[f1].clip(upper=20)) * df[f2]  # 指数乘积

    interaction_transforms_df = pd.DataFrame(interaction_transforms)

    print("特征交互完成")

    print('-------------------------------------------------------------------------------------------')

    print("开始基于聚类的特征生成")

    cluster_transforms = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        kmeans = KMeans(n_clusters=3, random_state=0)
        cluster_transforms['cluster'] = kmeans.fit_predict(df[numeric_cols])
        cluster_transforms['distance_to_centroid'] = np.linalg.norm(
            df[numeric_cols].values - kmeans.cluster_centers_[cluster_transforms['cluster']], axis=1
        )
    else:
        print("特征不足，无法进行聚类。")

    cluster_transforms_df = pd.DataFrame(cluster_transforms)

    print("基于聚类的特征生成完成")

    print('-------------------------------------------------------------------------------------------')

    print("开始非线性变换和组合")

    nonlinear_transforms = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        nonlinear_transforms[f'sqrt_{col}_mul_log_{col}'] = np.sqrt(df[col].clip(lower=0)) * np.log1p(df[col]) # 平方根和对数组合
        nonlinear_transforms[f'exp_{col}_plus_square_{col}'] = np.exp(df[col].clip(upper=20)) + df[col] ** 2 # 指数和平方组合
        nonlinear_transforms[f'sin_{col}_plus_cos_{col}'] = np.sin(df[col]) + np.cos(df[col]) # 正弦和余弦组合
        nonlinear_transforms[f'tan_{col}_mul_tanh_{col}'] = np.tan(df[col].clip(-np.pi / 2, np.pi / 2)) * np.tanh(
            df[col]) # 正切和双曲正切组合
        nonlinear_transforms[f'sqrt_{col}_plus_exp_{col}'] = np.sqrt(df[col].clip(lower=0)) + np.exp(
            df[col].clip(upper=20)) # 平方根和指数组合
        nonlinear_transforms[f'abs_{col}_plus_log_{col}'] = np.abs(df[col]) + np.log1p(df[col].replace(0, 1e-5)) # 绝对值和对数组合
        nonlinear_transforms[f'cube_{col}_plus_log_{col}'] = (df[col] ** 3) + np.log1p(df[col]) # 立方和对数组合
        nonlinear_transforms[f'square_{col}_mul_sqrt_{col}'] = (df[col] ** 2) * np.sqrt(df[col].clip(lower=0)) # 平方与平方根组合
        nonlinear_transforms[f'exp_{col}_minus_log_{col}'] = np.exp(df[col].clip(upper=20)) - np.log1p(
            df[col].replace(0, 1e-5)) # 指数减对数组合
        nonlinear_transforms[f'sqrt_{col}_plus_inverse_{col}'] = np.sqrt(df[col].clip(lower=0)) + (
                    1 / df[col].replace(0, 1e-5)) # 平方根与倒数组合

    nonlinear_transforms_df = pd.DataFrame(nonlinear_transforms)

    print("非线性变换和组合完成")

    # print('-------------------------------------------------------------------------------------------')
    #
    # print("开始使用FeatureTools生成特征")
    # df['index'] = df.index
    # es = ft.EntitySet(id='data')
    # es = es.add_dataframe(dataframe_name='data', dataframe=df, index='index')
    #
    # # 运行DFS生成特征
    # feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='data')
    # print("FeatureTools特征生成完成")


    data = pd.concat(
        [data, math_transforms_df, bin_transforms_df, interaction_transforms_df, cluster_transforms_df,
         nonlinear_transforms_df], axis=1)

    if 'index' in data.columns:
        data = data.drop(columns=['index'])

    return data

input_file = 'ILM_SNP_Train.csv'
df = pd.read_csv(input_file)

df = FE(df)

output_file = 'FE_ILM_SNP_Train.csv'
print("结果开始保存")
df.to_csv(output_file, index = False)
print("结果保存完成")     # 确认是否执行到此

print('新特征已生成并输出到:', output_file)