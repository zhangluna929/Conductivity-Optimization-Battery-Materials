import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import joblib
import random
import os
from tqdm import tqdm


# 🚩 设置全局随机种子，保证科研结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# 1️⃣ Load the dataset
required_cols = ['material_1', 'material_2', 'temperature', 'pressure', 'conductivity']
missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")


# 🚩 【缺失值检查】
print("Checking for missing values...")
print(data.isnull().sum())
if data.isnull().sum().sum() > 0:
    print("Missing values found, filling with column means (for numeric) and mode (for categorical).")
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].mean())

# 🚩 【异常值过滤】
def remove_outliers_iqr(df, cols, factor=1.5):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        after = len(df)
        print(f"Removed {before - after} outliers from '{col}'")
    return df

data = remove_outliers_iqr(data, ['temperature', 'pressure', 'conductivity'])

# 🚩 【特征工程自动处理】
feature_cols = ['material_1', 'material_2', 'temperature', 'pressure']
target_col = 'conductivity'

X = data[feature_cols]
y = data[target_col]

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# 🚩 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

# 🚩 Random Forest Pipeline + GridSearchCV
param_grid_rf = {
    'regressor__n_estimators': [50, 100, 200, 300],
    'regressor__max_depth': [5, 10, 20, 30, None],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2]
}

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=SEED))
])

from sklearn.model_selection import KFold

cv_strategy = KFold(n_splits=3, shuffle=True, random_state=SEED)


grid_search_rf = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid_rf,
        cv=cv_strategy,
    n_jobs=-1,
    verbose=0,
    scoring='neg_mean_squared_error'
)

for _ in tqdm(range(1), desc="Training Random Forest GridSearch"):
    grid_search_rf.fit(X_train, y_train)


print("Best Parameters for Random Forest:", grid_search_rf.best_params_)

y_pred_rf = grid_search_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"Random Forest - MSE: {mse_rf:.4f}, R2: {r2_rf:.4f}, MAE: {mae_rf:.4f}")

# 🚩 SVR Pipeline + GridSearchCV (补充公平对比)
param_grid_svr = {
    'regressor__C': [1, 10, 100],
    'regressor__epsilon': [0.01, 0.1, 0.2],
    'regressor__kernel': ['rbf', 'linear']
}

svr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR())
])

grid_search_svr = GridSearchCV(
    estimator=svr_pipeline,
    param_grid=param_grid_svr,
    cv=cv_strategy,
    n_jobs=-1,
    verbose=0,
    scoring='neg_mean_squared_error'
)

for _ in tqdm(range(1), desc="Training SVR GridSearch"):
    grid_search_svr.fit(X_train, y_train)


print("Best Parameters for SVR:", grid_search_svr.best_params_)

y_pred_svr = grid_search_svr.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
print(f"SVR - MSE: {mse_svr:.4f}, R2: {r2_svr:.4f}, MAE: {mae_svr:.4f}")

# 🚩 残差可视化（分析系统性偏差）
residuals_rf = y_test - y_pred_rf
residuals_svr = y_test - y_pred_svr

plt.figure(figsize=(10, 5))
plt.scatter(y_test, residuals_rf, color='red', label='RF Residuals', alpha=0.5)
plt.scatter(y_test, residuals_svr, color='blue', label='SVR Residuals', alpha=0.5)
plt.axhline(0, linestyle='--', color='k')
plt.xlabel('True Values (Conductivity)')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residuals vs True Values')
plt.show()


# 🚩 可视化预测结果对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, color='red', label='Random Forest', alpha=0.5)
plt.scatter(y_test, y_pred_svr, color='blue', label='SVR', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('True Values (Conductivity)')
plt.ylabel('Predictions')
plt.legend()
plt.title('Predictions vs True Values (Random Forest vs SVR)')
plt.show()

# 🚩 【特征重要性可视化】
best_rf_pipeline = grid_search_rf.best_estimator_
rf_model = best_rf_pipeline.named_steps['regressor']
ohe = best_rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(ohe_feature_names)

# 使用 DataFrame 保证排序稳定与可读性
importances_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(importances_df['feature'], importances_df['importance'], align='center')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 🚩 SHAP 可解释性分析（解释哪些特征影响最大）
import shap

# 对 Random Forest 模型进行解释
explainer = shap.TreeExplainer(rf_model)

# 获取预处理后的测试集特征用于解释
X_test_transformed = preprocessor.transform(X_test)

# 计算 SHAP 值（解释每个特征对预测结果的贡献）
shap_values = explainer.shap_values(X_test_transformed)

# 绘制 SHAP 总结图（特征影响力排序，可用于论文插图）
shap.summary_plot(shap_values, X_test_transformed, feature_names=all_feature_names)


# 🚩 保存模型
joblib.dump(grid_search_rf.best_estimator_, 'best_rf_pipeline.pkl')
joblib.dump(grid_search_svr.best_estimator_, 'best_svr_pipeline.pkl')
print("Pipelines saved successfully (Random Forest and SVR).")
joblib.dump(preprocessor, 'preprocessor.pkl')
print("Preprocessor saved successfully.")

# 绘图
results_df = pd.DataFrame({
    'True_Conductivity': y_test.values,
    'RF_Predicted': y_pred_rf,
    'SVR_Predicted': y_pred_svr
})
results_df.to_csv('prediction_results.csv', index=False)
print("Prediction results saved to prediction_results.csv")
