import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pmdarima.arima import auto_arima

# 准备时间序列数据
# 假设我们有一个名为"series_data"的Series对象，包含时间索引和对应的数值
# series_data = pd.Series(data, index=dates)
# 1. 数据准备
data = pd.read_csv('data\low_temperature\daily-min-temperatures.csv')  # 假设数据集保存为CSV文件
data = data[:608]
date = pd.to_datetime(data['date'])  # 将日期列转换为datetime类型

data['date'] = date
X = data[['date']]  # 特征列
y = data['Temp']  # 目标列
# 拆分训练集和测试集
# series_data = pd.Series(y,index=X)
# train_size = int(len(series_data) * 0.8)
# train_data, test_data = series_data[:train_size], series_data[train_size:]
y = pd.Series(y.values)
train_y = y[:-96]
# 创建ARIMA模型
print(train_y,type(train_y))

model = auto_arima(train_y, 
                   start_p=0, start_q=0,
                   max_p=5, max_q=5, m=12,
                   start_P=0, seasonal=True,
                   d=None, D=1, trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)

# # 拟合ARIMA模型
# model_fit = model.fit()

# 进行预测
forecast = model.predict(len(y[-96:]))

# 打印预测结果
print(y[-96:].values)
print(forecast.values,type(forecast))
X_data = [i for i in range(96)]
# 可视化预测结果
plt.plot(X_data, y[-96:].values, label='Actual')
plt.plot(X_data, forecast.values, label='Predicted')
plt.legend()
plt.show()