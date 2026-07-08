import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===================== 基础配置 =====================
INPUT_FILE = "清洗后双组分NPEF170_NPEL128配方表.xlsx"
# 6个输入特征
FEATURE_COLS = [
    "银含量",
    "银包铜粉_占比",
    "纳米银粉_占比",
    "粘结相1_占比",
    "粘结相2_占比",
    "固化剂_占比"
]
TARGET_COL = "体电阻率"
# 中文绘图设置
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 数据读取与清洗 =====================
# 读取完整表格
df = pd.read_excel(INPUT_FILE)
# 只保留特征+目标列参与建模
df_model = df[FEATURE_COLS + [TARGET_COL]].copy()
# 核心规则：任意特征/体电阻率为空，直接删除整行（不填充0）
df_model = df_model.dropna(subset=FEATURE_COLS + [TARGET_COL])
print(f"原始表格总行数：{len(df)}")
print(f"删除含空值样本后建模行数：{len(df_model)}")

# 划分X、y
X = df_model[FEATURE_COLS]
y = df_model[TARGET_COL]

# 划分训练集80%、测试集20%，固定随机种子保证复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ===================== 5类回归模型定义 =====================
model_dict = {
    "线性回归": LinearRegression(),
    "随机森林": RandomForestRegressor(random_state=42, max_depth=5, min_samples_leaf=4, n_estimators=100),
    "GBDT原生梯度提升树": GradientBoostingRegressor(random_state=42, max_depth=4, n_estimators=100),
    "XGBoost": XGBRegressor(random_state=42, max_depth=4, verbosity=0),
    "LightGBM": LGBMRegressor(random_state=42, max_depth=4, verbose=-1)
}

eval_results = []
best_r2 = -999
best_model = None
best_name = ""

# 循环训练所有模型
for name, model in model_dict.items():
    print(f"\n======== 开始训练 {name} ========")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 计算评价指标
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # 5折交叉验证R2
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

    print(f"测试集R²={r2:.4f} | 5折交叉验证R²={cv_r2:.4f}")
    print(f"MAE={mae:.3f} | RMSE={rmse:.3f}")

    eval_results.append({
        "模型名称": name,
        "测试R2": round(r2, 4),
        "交叉验证R2": round(cv_r2, 4),
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3)
    })

    # 更新最优模型
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

# ===================== 输出模型汇总结果 =====================
res_df = pd.DataFrame(eval_results).sort_values("测试R2", ascending=False)
print("\n===== 全部模型性能（按测试R2降序）=====")
print(res_df.to_string(index=False))
print(f"\n最优模型：{best_name}，最优测试R² = {best_r2:.4f}")

# 保存最优模型本地
joblib.dump(best_model, "双组分_6特征_最优电阻率模型.pkl")
print("最优模型文件已保存：双组分_6特征_最优电阻率模型.pkl")

# ===================== 绘图1：真实值vs预测值拟合图（已修复变量错误） =====================
y_best_pred = best_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_best_pred, alpha=0.7, s=70)
# 修复点：全部使用y_best_pred，删除不存在的y_best变量
minv = min(y_test.min(), y_best_pred.min())
maxv = max(y_test.max(), y_best_pred.max())
plt.plot([minv, maxv], [minv, maxv], "r--", linewidth=2, label="理想拟合线")
plt.title(f"{best_name} 体电阻率真实值-预测对比 R²={best_r2:.4f}")
plt.xlabel("真实体电阻率")
plt.ylabel("预测体电阻率")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("双组分_6特征_拟合效果图.png", dpi=300)
plt.show()

# ===================== 绘图2：特征重要度（线性回归跳过） =====================
if best_name != "线性回归":
    imp_data = pd.DataFrame({
        "特征": FEATURE_COLS,
        "重要度得分": best_model.feature_importances_
    }).sort_values("重要度得分", ascending=True)
    plt.figure(figsize=(12, 7))
    plt.barh(imp_data["特征"], imp_data["重要度得分"], color="#1f77b4")
    plt.title("六大特征对体电阻率的重要程度")
    plt.xlabel("特征重要度分数")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("双组分_6特征_重要性图.png", dpi=300)
    plt.show()

# ===================== 预测调用函数 =====================
def predict_resistivity(ag, agcu, nano, b1, b2, cure):
    """
    入参顺序对应特征：
    1.银含量 2.银包铜粉_占比 3.纳米银粉_占比
    4.粘结相1_占比 5.粘结相2_占比 6.固化剂_占比
    """
    input_data = pd.DataFrame([[ag, agcu, nano, b1, b2, cure]], columns=FEATURE_COLS)
    model = joblib.load("双组分_6特征_最优电阻率模型.pkl")
    pred_val = model.predict(input_data)[0]
    print(f"\n【输入配方参数】")
    print(f"银含量:{ag}  银包铜粉占比:{agcu}  纳米银粉占比:{nano}")
    print(f"粘结相1占比:{b1}  粘结相2占比:{b2}  固化剂占比:{cure}")
    print(f"预测体电阻率 = {pred_val:.4f} Ω·cm")
    return pred_val

# 使用示例
# predict_resistivity(52, 68, 22, 1.9, 1.1, 2.6)
