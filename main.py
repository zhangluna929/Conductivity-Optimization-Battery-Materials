import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import skew, kurtosis, zscore
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import joblib
import random
import os
import shap

# 种子SEED
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# 1.导数据

data = pd.read_csv("your_file.csv")

required_cols = ['material_1', 'material_2', 'temperature', 'pressure', 'conductivity']

available_cols = data.columns.tolist()

missing_cols = [col for col in required_cols if col not in available_cols]

if missing_cols:
    print(f"Warning: Missing columns in dataset: {missing_cols}")
else:
    print("All required columns are present.")

for col in missing_cols:
    if col in ['temperature', 'pressure']:  
        data[col] = 0  
    elif col in ['material_1', 'material_2']:  
        data[col] = 'Unknown'  

print("数据加载完成，继续处理...")

print(data.head())

# 2.缺失值检查
print("Checking for missing values...")
print(data.isnull().sum())
if data.isnull().sum().sum() > 0:
    print("Missing values found, filling with column means (for numeric) and mode (for categorical).")
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].mean())


# 3.异常值过滤
def select_outlier_method(df, cols):

    for col in cols:
        data = df[col]
        print(f"分析 {col} 列的数据类型和分布：")

        if data.dtype == 'object':
            print(f"{col} 是类别型数据，跳过异常值检测。")
            continue

        skewness = data.skew()
        kurtosis = data.kurtosis()

        print(f"偏态（Skewness）: {skewness:.2f}, 峰态（Kurtosis）: {kurtosis:.2f}")

        if abs(skewness) < 1 and abs(kurtosis) < 3:
            print(f"{col} 的数据接近正态分布，选择 Z-score 方法进行异常值检测。")
            remove_outliers_zscore(df, col, threshold=3)
        elif abs(skewness) > 1 and abs(kurtosis) > 3:
            print(f"{col} 偏态和峰态较高，选择 IQR 方法进行异常值检测。")
            remove_outliers_iqr(df, col)
        else:
            print(f"{col} 的数据特征适合使用 Isolation Forest 检测异常值。")
            remove_outliers_isolation_forest(df, col)


def remove_outliers_zscore(df, col, threshold=3):
    """Z-score检测异常值"""
    z_scores = zscore(df[col])  # 计算Z-score
    outliers = np.abs(z_scores) > threshold
    df = df[~outliers]
    print(f"使用 Z-score 去除 {outliers.sum()} 个异常值")
    return df


def remove_outliers_iqr(df, col, factor=1.5):
    """IQR检测异常值"""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]
    print(f"使用IQR去除 {len(df)} 个异常值")
    return df


def remove_outliers_isolation_forest(df, col):
    """Isolation Forest检测异常值"""
    model = IsolationForest(contamination=0.1)
    outliers = model.fit_predict(df[[col]])  # 返回-1为异常值
    df = df[outliers == 1]
    print(f"使用 Isolation Forest 去除 {outliers[outliers == -1].sum()} 个异常值")
    return df


def remove_outliers_dbscan(df, col):
    """使用 DBSCAN 检测异常值"""
    model = DBSCAN(eps=0.5, min_samples=5)  # DBSCAN参数
    outliers = model.fit_predict(df[[col]])
    df = df[outliers != -1]
    print(f"使用 DBSCAN 去除 {outliers[outliers == -1].sum()} 个异常值")
    return df

data = remove_outliers_iqr(data, ['temperature', 'pressure', 'conductivity'])

# 4.特征工程自动处理
feature_cols = ['material_1', 'material_2', 'temperature', 'pressure']
target_col = 'conductivity'

X = data[feature_cols]  
y = data[target_col]  

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

def select_scaling_method(df, cols):
    for col in cols:
        data = df[col]
        skewness = skew(data)
        kurt = kurtosis(data)

        if abs(skewness) < 1 and abs(kurt) < 3:
            df[col] = StandardScaler().fit_transform(data.values.reshape(-1, 1))
        elif abs(skewness) > 1 and abs(kurt) > 3:
            df[col] = RobustScaler().fit_transform(data.values.reshape(-1, 1))
        else:
            df[col] = MinMaxScaler().fit_transform(data.values.reshape(-1, 1))
    return df

X = select_scaling_method(X, numerical_cols)

def select_category_encoding(df, cols):
    encoders = {}
    for col in cols:
        unique_vals = df[col].nunique()
        print(f"分析 {col} 列的唯一值个数：{unique_vals}")

        if unique_vals > 10:  # 超过10个唯一值，为无序类别，用OneHotEncoder
            encoders[col] = OneHotEncoder(handle_unknown='ignore')
            print(f"{col} 是无序类别，使用 OneHotEncoder。")
        else:  # 少于10个唯一值，为有序类别，使用 OrdinalEncoder
            encoders[col] = OrdinalEncoder()
            print(f"{col} 是有序类别，使用 OrdinalEncoder。")

    return encoders

category_encoders = select_category_encoding(X, categorical_cols)

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', Pipeline(steps=[('encoder', category_encoders[col]) for col in categorical_cols]), categorical_cols)
])

# 5.划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

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

# 6. 残差可视化
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

# 7.结果对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, color='red', label='Random Forest', alpha=0.5)
plt.scatter(y_test, y_pred_svr, color='blue', label='SVR', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('True Values (Conductivity)')
plt.ylabel('Predictions')
plt.legend()
plt.title('Predictions vs True Values (Random Forest vs SVR)')
plt.show()

# 8.特征重要性可视化
best_rf_pipeline = grid_search_rf.best_estimator_
rf_model = best_rf_pipeline.named_steps['regressor']
ohe = best_rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(ohe_feature_names)

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

# 9.SHAP 可解释性分析
explainer = shap.TreeExplainer(rf_model)
X_test_transformed = preprocessor.transform(X_test)
shap_values = explainer.shap_values(X_test_transformed)
shap.summary_plot(shap_values, X_test_transformed, feature_names=all_feature_names)

# 10.保存
joblib.dump(grid_search_rf.best_estimator_, 'best_rf_pipeline.pkl')
joblib.dump(grid_search_svr.best_estimator_, 'best_svr_pipeline.pkl')
print("Pipelines saved successfully (Random Forest and SVR).")
joblib.dump(preprocessor, 'preprocessor.pkl')
print("Preprocessor saved successfully.")

# 11.绘图
results_df = pd.DataFrame({
    'True_Conductivity': y_test.values,
    'RF_Predicted': y_pred_rf,
    'SVR_Predicted': y_pred_svr
})
results_df.to_csv('prediction_results.csv', index=False)
print("Prediction results saved to prediction_results.csv")
