import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import joblib

#1: Load the dataset
#请结合自己的CSV原始数据，包含电池材料的电导率数据
data = pd.read_csv('battery_materials_conductivity.csv')

# 打印出数据的前几行，看看数据结构
print(data.head())

#2: Data Preprocessing
# 数据中的 'material_1', 'material_2', 'temperature', 'pressure' 是特征
# 'conductivity' 是我们需要预测的目标变量
X = data[['material_1', 'material_2', 'temperature', 'pressure']]  # 特征
y = data['conductivity']  # 目标变量

#3: Feature Scaling (Standardization)
# 对特征进行标准化处理，确保数据的尺度统一
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#4: Split the data into training and testing sets
# 将数据划分为训练集和测试集，测试集比例为20%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#5: Train the model
# 使用支持向量回归（SVR）和随机森林回归（Random Forest）

#5.1 Support Vector Regression (SVR) Model
# 训练 SVR 模型来进行预测
svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_model.fit(X_train, y_train)

#6: Evaluate the SVR model
# 用测试集评估 SVR 模型的效果
y_pred_svr = svr_model.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f"SVR Model - Mean Squared Error: {mse_svr}")

#7: Hyperparameter Tuning for Random Forest using GridSearchCV
# 对随机森林回归器进行超参数调优
param_grid = {
    'n_estimators': [100, 200],  # 森林中树的数量
    'max_depth': [10, 20, None],  # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 拆分节点所需的最小样本数
    'min_samples_leaf': [1, 2, 4]   # 叶子节点的最小样本数
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 打印出最优的随机森林超参数
print("Best Parameters for Random Forest:", grid_search.best_params_)

#8: Evaluate the Random Forest model
# 用最优参数训练后的随机森林模型对测试集进行评估
y_pred_rf = grid_search.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest Model - Mean Squared Error: {mse_rf}")

#9: Visualize the predictions
# 用 Matplotlib 可视化 SVR 和 Random Forest 的预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_svr, color='blue', label='SVR Predictions', alpha=0.6)
plt.scatter(y_test, y_pred_rf, color='red', label='Random Forest Predictions', alpha=0.6)
plt.xlabel('True Values (Conductivity)')
plt.ylabel('Predictions')
plt.legend()
plt.title('SVR vs Random Forest Predictions')
plt.show()

#10: Save the model for future use (optional)
# 如果后续需要使用已训练好的模型，可以将其保存为文件
joblib.dump(grid_search.best_estimator_, 'best_rf_model.pkl')
