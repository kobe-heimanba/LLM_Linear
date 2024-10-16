import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
from datetime import datetime, timedelta

# 1. 数据准备
data = pd.read_csv('..\\data\\ETT\\ETTh1.csv')  # 假设数据集保存为CSV文件
data = data[:608]
date = pd.to_datetime(data['date'])  # 将日期列转换为datetime类型

data['date'] = date
X = data[['date']]  # 特征列
y = data['OT']  # 目标列

# 2. 数据预处理和特征工程
# 进行数据预处理和特征工程的步骤，如处理缺失值、特征转换等
# start_date = datetime(2024, 5, 1)
# date = [start_date + timedelta(days=i) for i in range(len(data))]
# date = pd.to_datetime(date)
# data['date'] = date
# X = data[['date']]
# 3. 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train,type(X_train))
print(X_test)
# X_train, X_test, y_train, y_test = X[:int(len(data)*0.8)], X[int(len(data)*0.8):],y[:int(len(data)*0.8)],y[int(len(data)*0.8):]
# print(X_train,type(X_train))
# model = pm.auto_arima(y_train,trend = 'c',seasonal = True, 
#                        seasonal_test= 'ocsb',
#                        stepwise=False,
#                        n_jobs = -1 ) #模型初始化,什么都默认
# print('bestmodel',model)
# y_pred = model.predict(10) #预测未来n_periods期

# print(y_pred)


# print(X_train)
# model = GradientBoostingRegressor()
# model.fit(X_train,y_train.values)
# y_pred = model.predict(X_test)
# print(X_train,type(X_train))
# print(y_train.values)
# 4. 模型训练
gbrt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbrt.fit(X_train, y_train)

# # 5. 模型评估
y_pred = gbrt.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print(y_pred,'\n')
print(y_test)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Coefficient of Determination (R^2):", r2)

# # # 6. 模型预测
# # new_data = pd.read_csv('new_data.csv')  # 新的未知数据
# # new_X = new_data[['Feature1', 'Feature2', ...]]  # 特征列
# # new_y_pred = gbrt.predict(new_X)

# 绘制实际观测值和预测值比较图
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_pred)), y_test.values, color='b', label='Actual')
plt.plot(range(len(y_pred)), y_pred, color='r', label='Predicted')
plt.xlabel('Data Points')
plt.ylabel('Temperature')
plt.title('Actual vs Predicted Temperature')
plt.legend()
plt.show()