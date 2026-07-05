# -*- coding: utf-8 -*-
"""
data_clean：56个HJT银包铜浆料配方数据分析
优化点：自动清除NO-BREAK SPACE特殊空格，消除字体缺失警告
适配动态材料品类，同时做数值占比分析+不同材料性能对比
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ========== 基础配置 ==========
INPUT_FILE = "peifang.xlsx"
# 更换微软雅黑优先，兼容特殊空白字符，消除Glyph160警告
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


# ========== 第一步：重构原始数据 + 清除特殊空格 ==========
def restructure_raw_data(file_path):
    print("=" * 60)
    print("第1步：重构原始数据，3行格式转一行结构化表")
    print("=" * 60)

    df_raw = pd.read_excel(file_path, header=None)
    # 清除Unicode 160不间断空格，彻底消除绘图警告
    df_raw = df_raw.replace(chr(160), "")
    # 替换所有"/"空标记为NaN
    df_raw = df_raw.replace(["/", " / ", "/ ", " /", "", " "], np.nan)

    # 固定组分、性能列名
    component_names = [
        "粘结相1", "粘结相2", "粘结相3", "稀释剂", "固化剂",
        "溶剂1", "溶剂2", "溶剂3", "触变剂", "分散剂", "交联剂",
        "银包铜粉", "微米银粉", "纳米银粉"
    ]
    performance_names = ["体电阻率", "粘度", "细度", "银含量"]

    recipes = []
    current_recipe = None

    for idx, row in df_raw.iterrows():
        first_col = str(row[0]).strip()

        # 识别配方开头（1号、2号...）
        if "号" in first_col and "品类" not in first_col and "占比" not in first_col:
            if current_recipe is not None:
                recipes.append(current_recipe)
            current_recipe = {"配方号": first_col}

        # 品类行：组分材料名 + 性能数值
        elif first_col == "品类" and current_recipe is not None:
            # 填充各组分材料名称
            for i, comp in enumerate(component_names):
                val = row[i + 1]
                current_recipe[f"{comp}_材料"] = val if pd.notna(val) else np.nan
            # 填充性能指标数值
            for i, perf in enumerate(performance_names):
                val = row[i + 15]
                if pd.notna(val):
                    try:
                        current_recipe[perf] = float(val)
                    except:
                        current_recipe[perf] = np.nan
                else:
                    current_recipe[perf] = np.nan

        # 占比行：各组分质量占比数值
        elif first_col == "占比" and current_recipe is not None:
            for i, comp in enumerate(component_names):
                val = row[i + 1]
                if pd.notna(val):
                    try:
                        current_recipe[f"{comp}_占比"] = float(val)
                    except:
                        current_recipe[f"{comp}_占比"] = np.nan
                else:
                    current_recipe[f"{comp}_占比"] = np.nan

    # 存入最后一组配方
    if current_recipe is not None:
        recipes.append(current_recipe)

    df = pd.DataFrame(recipes)
    print(f"✅ 重构完成：共提取 {len(df)} 个配方")
    print(f"   组分数：{len(component_names)} 个（材料+占比）")
    print(f"   性能指标：{len(performance_names)} 个")
    return df


# ========== 第二步：数值占比与性能相关性分析 ==========
def numeric_analysis(df):
    print("\n" + "=" * 60)
    print("第2步：数值占比 vs 性能 趋势分析")
    print("=" * 60)

    # 提取占比列、性能列
    ratio_cols = [col for col in df.columns if "_占比" in col]
    perf_cols = ["体电阻率", "粘度", "细度", "银含量"]
    perf_cols = [c for c in perf_cols if c in df.columns]

    # 筛选有效数值列（至少5个有效数据）
    valid_ratio_cols = []
    for col in ratio_cols:
        if df[col].notna().sum() >= 5:
            valid_ratio_cols.append(col)

    all_numeric = valid_ratio_cols + perf_cols
    print(f"有效占比列：{len(valid_ratio_cols)} 个")
    print(f"性能指标列：{perf_cols}")

    # 图1：相关性热力图
    print("\n📊 生成相关性热力图...")
    corr_matrix = df[all_numeric].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0,
                fmt=".2f", linewidths=0.5, annot_kws={"size": 9})
    plt.title("各组分占比与性能相关性热力图", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("data_clean_图1_占比性能相关性热力图.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 已生成：data_clean_图1_占比性能相关性热力图.png")

    # 输出影响电阻率Top5核心因素
    if "体电阻率" in corr_matrix.columns:
        resist_corr = corr_matrix["体电阻率"].abs().sort_values(ascending=False)
        print("\n【核心结论】对体电阻率影响最大Top5因素：")
        count = 0
        for col, val in resist_corr.items():
            if col != "体电阻率" and count < 5:
                direct = "负相关" if corr_matrix["体电阻率"][col] < 0 else "正相关"
                print(f"  {count + 1}. {col.replace('_占比', '')}：相关系数 {val:.3f}（{direct}）")
                count += 1

    # 图2：Top5关键组分-电阻率散点趋势图
    print("\n📊 生成关键组分散点趋势图...")
    target = "体电阻率" if "体电阻率" in df.columns else perf_cols[0]
    top_features = resist_corr.head(6).index.tolist()
    top_features = [f for f in top_features if f != target]
    plot_count = 0
    for feat in top_features[:5]:
        valid_data = df[[feat, target]].dropna()
        if len(valid_data) < 5:
            continue
        plt.figure(figsize=(8, 6))
        plt.scatter(valid_data[feat], valid_data[target], alpha=0.7, s=70, edgecolors="white")
        # 二次拟合趋势线
        try:
            z = np.polyfit(valid_data[feat], valid_data[target], 2)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data[feat].min(), valid_data[feat].max(), 100)
            plt.plot(x_line, p(x_line), "r--", linewidth=2, label="趋势线")
            plt.legend()
        except Exception:
            pass
        plt.xlabel(feat.replace("_占比", ""), fontsize=12)
        plt.ylabel(target, fontsize=12)
        plt.title(f"{feat.replace('_占比', '')}占比 vs {target}", fontsize=14, pad=15)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"data_clean_图2_{plot_count + 1}_{feat.replace('_占比', '')}vs电阻率.png", dpi=300, bbox_inches="tight")
        plt.close()
        plot_count += 1
    print(f"✅ 已生成：{plot_count} 张关键组分散点趋势图")

    # 图3：体电阻率分布直方图
    print("\n📊 生成体电阻率分布直方图...")
    if "体电阻率" in df.columns:
        valid_resist = df["体电阻率"].dropna()
        if len(valid_resist) >= 5:
            plt.figure(figsize=(8, 6))
            plt.hist(valid_resist, bins=12, color="#4c72b0", alpha=0.7, edgecolor="white")
            mean_val = valid_resist.mean()
            median_val = valid_resist.median()
            plt.axvline(mean_val, color="red", linestyle="--", linewidth=2,
                        label=f"平均值：{mean_val:.2f}")
            plt.axvline(median_val, color="green", linestyle="--", linewidth=2,
                        label=f"中位数：{median_val:.2f}")
            plt.xlabel("体电阻率", fontsize=12)
            plt.ylabel("配方数量", fontsize=12)
            plt.title(f"{len(valid_resist)}个配方体电阻率分布", fontsize=14, pad=15)
            plt.legend()
            plt.grid(alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig("data_clean_图3_体电阻率分布直方图.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("✅ 已生成：data_clean_图3_体电阻率分布直方图.png")

    # 输出数值统计汇总表
    stats_df = df[all_numeric].describe().T
    stats_df = stats_df[["count", "mean", "50%", "min", "max", "std"]]
    stats_df.columns = ["有效样本数", "平均值", "中位数", "最小值", "最大值", "标准差"]
    stats_df.to_excel("data_clean_数值参数统计汇总表.xlsx")
    print("✅ 已生成：data_clean_数值参数统计汇总表.xlsx")


# ========== 第三步：不同材料品类性能对比分析 ==========
def material_category_analysis(df):
    print("\n" + "=" * 60)
    print("第3步：不同材料品类 vs 电阻率 对比分析")
    print("=" * 60)

    material_cols = [col for col in df.columns if "_材料" in col]
    material_stat_dict = {}
    # 统计各组分材料出现频次
    print("\n【各组分材料品类统计】")
    for col in material_cols:
        mat_list = df[col].dropna().tolist()
        mat_list = [m for m in mat_list if str(m).strip() != ""]
        if len(mat_list) > 0:
            counter = Counter(mat_list)
            material_stat_dict[col] = counter
            comp_name = col.replace("_材料", "")
            print(f"\n{comp_name}（共{len(counter)}种材料）：")
            for mat, cnt in counter.most_common(5):
                print(f"  · {mat}：{cnt} 个配方")

    # 绘制不同材料箱线对比图
    target_perf = "体电阻率" if "体电阻率" in df.columns else None
    plot_count = 0
    if target_perf is not None:
        print("\n📊 生成材料品类性能对比图...")
        for col in material_cols:
            comp_name = col.replace("_材料", "")
            valid_data = df[[col, target_perf]].dropna()
            mat_count = valid_data[col].value_counts()
            valid_mats = mat_count[mat_count >= 2].index.tolist()
            if len(valid_mats) < 2:
                continue
            plot_data = valid_data[valid_data[col].isin(valid_mats)]
            plt.figure(figsize=(max(8, len(valid_mats) * 1.5), 6))
            sns.boxplot(x=col, y=target_perf, data=plot_data)
            plt.xlabel(f"{comp_name}材料型号", fontsize=12)
            plt.ylabel("体电阻率", fontsize=12)
            plt.title(f"不同{comp_name}材料的体电阻率对比", fontsize=14, pad=15)
            plt.xticks(rotation=15)
            plt.grid(alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(f"data_clean_图4_{plot_count + 1}_{comp_name}材料性能对比.png", dpi=300, bbox_inches="tight")
            plt.close()
            plot_count += 1
        print(f"✅ 已生成：{plot_count} 张材料品类性能对比图")

    # 输出材料品类统计Excel
    with pd.ExcelWriter("data_clean_材料品类统计汇总.xlsx") as writer:
        for col, counter in material_stat_dict.items():
            sheet_name = col.replace("_材料", "")[:30]
            pd.DataFrame(counter.most_common(), columns=["材料名称", "出现次数"]).to_excel(
                writer, sheet_name=sheet_name, index=False
            )
    print("✅ 已生成：data_clean_材料品类统计汇总.xlsx")


# ========== 程序入口 ==========
if __name__ == "__main__":
    # 1、重构原始配方数据
    df_formula = restructure_raw_data(INPUT_FILE)
    # 保存结构化中间表，feature_engineering/model_train直接读取使用
    df_formula.to_excel("data_clean_重构后_配方结构化表.xlsx", index=False)
    print("💾 已保存结构化配方表：data_clean_重构后_配方结构化表.xlsx")

    # 2、数值占比与性能分析
    numeric_analysis(df_formula)
    # 3、材料品类对比分析
    material_category_analysis(df_formula)

    print("\n" + "=" * 60)
    print("🎉 data_clean全部分析流程执行完成！")
    print("📁 全部输出文件清单：")
    print("  1. data_clean_重构后_配方结构化表.xlsx（feature_engineering/model_train直接读取）")
    print("  2. data_clean_图1_占比性能相关性热力图.png")
    print("  3. data_clean_图2_关键组分散点趋势图（多张）")
    print("  4. data_clean_图3_体电阻率分布直方图.png")
    print("  5. data_clean_图4_不同材料性能对比箱线图（多张）")
    print("  6. data_clean_数值参数统计汇总表.xlsx")
    print("  7. data_clean_材料品类统计汇总.xlsx")
    print("=" * 60)