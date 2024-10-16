import numpy as np
import torch
import torch.nn as nn
class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, pred, true):
        epsilon = 1e-7  # 用于避免分母为零的情况
        numerator = torch.abs(true - pred)
        denominator = (torch.abs(true) + torch.abs(pred)) / 2
        smape_loss = torch.mean(numerator / (denominator + epsilon))*200
        return smape_loss

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))*100


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def MRAE(pred, true):
    """
    计算MRAE（Mean Relative Absolute Error）
    参数：
      - actual: 真实值数组
      - forecast: 预测值数组
    返回值：
      MRAE评价指标的值
    """
    # true = np.array(true.cpu())
    # pred = np.array(pred.detach().numpy())
    return np.mean(np.abs(pred - true) / np.mean(true))*100 

def SMAPE(pred,true):
    """
    计算SMAPE（Symmetric Mean Absolute Percentage Error）
    参数：
      - true: 真实值数组
      - pred: 预测值数组
    返回值：
      SMAPE评价指标的值
    """
    # true = np.array(true.cpu())
    # pred = np.array(pred.cpu().detach().numpy())
    
    return np.mean(np.abs(pred - true) / (np.abs(pred) + np.abs(true)))*200

def mase(pred, true, training_data):
    """
    计算MASE（Mean Absolute Scaled Error）
    参数：
      - true: 真实值数组
      - pred: 预测值数组
      - training_data: 训练集真实值数组
    返回值：
      MASE评价指标的值
    """
    true = np.array(true.cpu())
    pred = np.array(pred.detach().numpy())
    training_data = np.array(training_data)
    scale = np.mean(np.abs(training_data[1:] - training_data[:-1]))
    return np.mean(np.abs(true - pred)) / scale

def owa(smape_value, mase_value, weights=(0.5, 0.5)):
    """
    计算OWA（Overall Weighted Average）
    参数：
      - smape_value: SMAPE评价指标的值
      - mase_value: MASE评价指标的值
      - weights: 权重元组，默认为(0.5, 0.5)
    返回值：
      OWA评价指标的值
    """
    smape_weight, mase_weight = weights
    return smape_weight * smape_value + mase_weight * mase_value

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    smape = SMAPE(pred,true)
    mape = MAPE(pred, true)
    mrae = MRAE(pred,true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse,smape, mape,mrae, mspe, rse, corr
