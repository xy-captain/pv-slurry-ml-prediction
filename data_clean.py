import pandas as pd
import numpy as np

# -------------------------- 配置项（修改为你的本地路径） --------------------------
# 原始配方表路径
input_file_path = r"C:\Users\Lenovo\Desktop\pl_patent_ai\项目3\项目3_重构后_配方结构化表.xlsx"
# 清洗后文件输出路径
output_file_path = r"C:\Users\Lenovo\Desktop\pl_patent_ai\项目3\双组分体系\清洗后双组分NPEF170_NPEL128配方表.xlsx"
# 目标筛选条件
target_binder1 = "环氧树脂NPEF170"
target_binder2 = "环氧树脂NPEL128"


# -------------------------------------------------------------------------------------

def clean_formula_data():
    print("===== 开始配方数据清洗 =====")

    # 1. 读取Excel文件
    df = pd.read_excel(input_file_path, sheet_name='Sheet1')
    print(f"原始数据总行数: {len(df)}")

    # 2. 全量空格清洗：列名+所有字符串单元格去除首尾空格
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("", np.nan)
            df[col] = df[col].replace("nan", np.nan)

    # 3. 删除全空行
    df = df.dropna(how='all').reset_index(drop=True)
    print(f"删除全空行后剩余行数: {len(df)}")

    # 4. 核心筛选：严格匹配双组分条件
    cond1 = df["粘结相1_材料"] == target_binder1
    cond2 = df["粘结相2_材料"] == target_binder2
    cond3 = df["粘结相3_材料"].isna()
    final_filter = cond1 & cond2 & cond3
    filtered_df = df[final_filter].reset_index(drop=True)
    print(f"符合筛选条件的配方数量: {len(filtered_df)}")

    # 5. 保存结果
    filtered_df.to_excel(output_file_path, index=False)
    print(f"✅ 清洗完成！结果已保存至: {output_file_path}")
    print("\n===== 前10行数据预览 =====")
    print(filtered_df[["配方号", "粘结相1_材料", "粘结相2_材料", "体电阻率", "银含量"]].head(10))


if __name__ == "__main__":
    clean_formula_data()
