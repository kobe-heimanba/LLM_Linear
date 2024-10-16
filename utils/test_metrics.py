import numpy as np

def smape(y_pred, y_true):
    """
    计算SMAPE（Symmetric Mean Absolute Percentage Error）
    """
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 200

def mrae(y_pred, y_true):
    """
    计算MRAE（Mean Relative Absolute Error）
    """
    return np.mean(np.abs(y_pred - y_true) / np.mean(y_true)) * 100

def mape(y_pred, y_true):
    """
    计算MAPE（Mean Absolute Percentage Error）
    """
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

# 示例用法
y_pred = np.array([12, 18, 32, 41, 48])
y_true = np.array([10, 20, 30, 40, 50])

print("SMAPE:", smape(y_pred, y_true))
print("MRAE:", mrae(y_pred, y_true))
print("MAPE:", mape(y_pred, y_true))