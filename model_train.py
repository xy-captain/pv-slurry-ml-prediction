# -*- coding: utf-8 -*-
"""
model_train：浆料配方多目标回归建模
输入：项目4_最终建模特征表.xlsx
输出：最优模型文件、模型评估图表、特征重要性图
预测目标：体电阻率、银含量
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]
INPUT = "项目4_最终建模特征表.xlsx"

# 1.读取数据，拆分特征X与预测标签y
df = pd.read_excel(INPUT)
# 两个预测目标
y_cols = ["体电阻率", "银含量"]
y = df[y_cols]
X = df.drop(columns=y_cols)
feat_names = X.columns.tolist()

# 划分训练集80%、测试集20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2.特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3.定义候选模型
models = {
    "线性回归": LinearRegression(),
    "随机森林": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42)
}

# 4.批量训练、评估所有模型
eval_result = []
best_model = None
best_r2 = -99
best_name = ""

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    eval_result.append({
        "模型": name,
        "R2": round(r2,4),
        "MAE": round(mae,3),
        "RMSE": round(rmse,3)
    })
    # 记录最优模型
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

# 输出评估表格
eval_df = pd.DataFrame(eval_result)
print("===== 各模型测试集评估指标 =====")
print(eval_df)
print(f"\n最优模型：{best_name}，测试集R2={best_r2:.4f}")

# 保存最优模型与标准化器
joblib.dump(best_model, "最优浆料预测模型.pkl")
joblib.dump(scaler, "特征标准化器.pkl")
print("模型已保存：最优浆料预测模型.pkl")

# 5.绘制特征重要性（以随机森林/XGB为例）
if hasattr(best_model, "feature_importances_"):
    imp = best_model.feature_importances_
    imp_df = pd.DataFrame({"特征":feat_names, "重要度":imp})
    imp_df = imp_df.sort_values("重要度", ascending=False).head(12)
    plt.figure(figsize=(10,6))
    plt.barh(imp_df["特征"], imp_df["重要度"])
    plt.gca().invert_yaxis()
    plt.title(f"{best_name} 前12重要特征")
    plt.xlabel("特征重要性权重")
    plt.tight_layout()
    plt.savefig("特征重要性分布图.png", dpi=150)
    plt.show()