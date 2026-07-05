# -*- coding: utf-8 -*-
"""
feature_engineering：HJT银包铜浆料配方 特征工程
输入：data_clean_重构后_配方结构化表.xlsx
输出：feature_engineering_最终建模特征表.xlsx（可直接用于model_train建模）
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings("ignore")

# ========== 配置 ==========
INPUT_FILE = "data_clean_重构后_配方结构化表.xlsx"
OUTPUT_FILE = "feature_engineering_最终建模特征表.xlsx"


# ========== 第一步：读取数据 + 缺失值填充 ==========
def fill_missing_values(df):
    print("=" * 60)
    print("第1步：缺失值统一填充处理")
    print("=" * 60)

    # 区分两类列：数值占比列、材料名称列
    ratio_cols = [col for col in df.columns if "_占比" in col]
    material_cols = [col for col in df.columns if "_材料" in col]

    # 数值占比空值 = 没添加该组分，填充为0
    df[ratio_cols] = df[ratio_cols].fillna(0)
    # 材料名称空值 = 未使用该组分，填充为“未添加”
    df[material_cols] = df[material_cols].fillna("未添加")

    print(f"数值占比列：{len(ratio_cols)} 个，空值已填充为0")
    print(f"材料名称列：{len(material_cols)} 个，空值已填充为'未添加'")
    return df


# ========== 第二步：构造工艺衍生特征（行业经验融入） ==========
def build_derived_features(df):
    print("\n" + "=" * 60)
    print("第2步：构造工艺衍生特征")
    print("=" * 60)

    # 1. 总粘结相（树脂）占比
    df["总粘结相占比"] = df["粘结相1_占比"] + df["粘结相2_占比"] + df["粘结相3_占比"]

    # 2. 总银系导电填料占比
    df["总银粉占比"] = df["银包铜粉_占比"] + df["微米银粉_占比"] + df["纳米银粉_占比"]

    # 3. 粉胶比（导电填料 / 树脂基体，浆料导电性能核心指标）
    df["粉胶比"] = np.where(df["总粘结相占比"] > 0,
                            df["总银粉占比"] / df["总粘结相占比"], 0)

    # 4. 纳米银占总银粉比例
    df["纳米银占比_总银"] = np.where(df["总银粉占比"] > 0,
                                     df["纳米银粉_占比"] / df["总银粉占比"], 0)

    # 5. 银铜质量比
    df["银铜质量比"] = np.where(df["银包铜粉_占比"] > 0,
                                df["纳米银粉_占比"] / df["银包铜粉_占比"], 0)

    # 6. 固化剂占树脂比例
    df["固化剂树脂比"] = np.where(df["总粘结相占比"] > 0,
                                  df["固化剂_占比"] / df["总粘结相占比"], 0)

    # 7. 总溶剂占比
    df["总溶剂占比"] = df["溶剂1_占比"] + df["溶剂2_占比"] + df["溶剂3_占比"]

    # 8. 总助剂占比（触变剂+分散剂+交联剂）
    df["总助剂占比"] = df["触变剂_占比"] + df["分散剂_占比"] + df["交联剂_占比"]

    print("已构造8个工艺衍生特征：")
    print("  总粘结相占比、总银粉占比、粉胶比、纳米银占比_总银")
    print("  银铜质量比、固化剂树脂比、总溶剂占比、总助剂占比")
    return df


# ========== 第三步：分类材料编码（文字转数字） ==========
def encode_material_features(df):
    print("\n" + "=" * 60)
    print("第3步：材料品类独热编码（文字转数值特征）")
    print("=" * 60)

    material_cols = [col for col in df.columns if "_材料" in col]
    encoded_dfs = []

    for col in material_cols:
        # 只保留出现次数≥2的材料，稀有材料归为“其他”，避免维度爆炸
        value_counts = df[col].value_counts()
        keep_materials = value_counts[value_counts >= 2].index.tolist()
        df[col] = df[col].apply(lambda x: x if x in keep_materials else "其他")

        # 独热编码
        encoder = OneHotEncoder(sparse=False, drop=None)
        encoded_arr = encoder.fit_transform(df[[col]])
        encoded_df = pd.DataFrame(
            encoded_arr,
            columns=[f"{col}_{cat}" for cat in encoder.categories_[0]]
        )
        encoded_dfs.append(encoded_df)

    # 拼接到原数据
    df_encoded = pd.concat([df.reset_index(drop=True)] + encoded_dfs, axis=1)
    # 删除原始文字材料列
    df_encoded = df_encoded.drop(columns=material_cols)

    print(f"材料列编码完成，新增 {sum([d.shape[1] for d in encoded_dfs])} 个材料编码特征")
    return df_encoded


# ========== 第四步：特征筛选（去掉无效、冗余特征） ==========
def filter_features(df):
    print("\n" + "=" * 60)
    print("第4步：特征筛选，剔除无效冗余特征")
    print("=" * 60)

    # 保留标签列（预测目标）
    label_cols = ["体电阻率", "粘度", "细度", "银含量"]
    label_cols = [c for c in label_cols if c in df.columns]

    # 1. 删除配方号（非特征）
    df = df.drop(columns=["配方号"], errors="ignore")

    # 2. 删除方差为0的特征（所有配方都一样，没有区分度）
    feature_cols = [col for col in df.columns if col not in label_cols]
    std_zero_cols = []
    for col in feature_cols:
        if df[col].std() == 0:
            std_zero_cols.append(col)
    df = df.drop(columns=std_zero_cols)

    # 3. 删除高度相关冗余特征（相关系数>0.95，保留一个即可）
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper.columns if any(upper[column] > 0.95)]
    df = df.drop(columns=high_corr_cols)

    print(f"删除无方差无效特征：{len(std_zero_cols)} 个")
    print(f"删除高度相关冗余特征：{len(high_corr_cols)} 个")
    print(f"最终保留特征数量：{len([c for c in df.columns if c not in label_cols])} 个")
    print(f"保留预测标签：{label_cols}")
    return df


# ========== 主程序 ==========
if __name__ == "__main__":
    # 读取data_clean输出的结构化表
    df_raw = pd.read_excel(INPUT_FILE)
    print(f"读取原始结构化数据：{df_raw.shape[0]} 个配方，{df_raw.shape[1]} 个字段")

    # 1. 缺失值填充
    df = fill_missing_values(df_raw)
    # 2. 构造衍生特征
    df = build_derived_features(df)
    # 3. 材料编码
    df = encode_material_features(df)
    # 4. 特征筛选
    df_final = filter_features(df)

    # 保存最终建模特征表
    df_final.to_excel(OUTPUT_FILE, index=False)
    print("\n" + "=" * 60)
    print(f"🎉 特征工程完成，最终文件已保存：{OUTPUT_FILE}")
    print(f"   数据规模：{df_final.shape[0]} 行样本，{df_final.shape[1]} 列（含特征+标签）")
    print("   可直接用于model_train机器学习建模")
    print("=" * 60)