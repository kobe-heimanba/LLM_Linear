import numpy as np
import matplotlib
import argparse
import matplotlib.pyplot as plt
import torch
from utils.tools import series_decomp,visual
from data_provider.data_factory import data_provider
import os
from pathlib import Path

def get_mse(y_pred,y_true):
    MSE = torch.mean((y_pred - y_true)**2)
    MSE = np.round(MSE.item(),4)
    return MSE


def visual_few():
    folder_path = r'./results_few/general/'
    """
    Trend results visualization
    """

    data_FiLM = np.load(folder_path+'FiLM_pred.npy')
    data_LlaLinear = np.load(folder_path+'LlaLinear_pred.npy')
    data_Nonstationary = np.load(folder_path+'Nonstationary_pred.npy')
    data_PatchTST = np.load(folder_path+'PatchTST_pred.npy')
    data_true = np.load(folder_path+'true.npy')
    data_x = np.load(folder_path+'x.npy')
    x = np.arange(0, 144)
    # print(data_FiLM.shape,data_true.shape)
    plt.figure(figsize=(11,4))
    plt.xlim(0,144)
    plt.plot(x,np.concatenate((data_x[80][-48:],data_FiLM[80]),axis=0).flatten(),color='#00FA9A', label='FiLM',linewidth=1.2,linestyle='--')
    plt.plot(x,np.concatenate((data_x[80][-48:],data_Nonstationary[80]),axis=0).flatten(),color='#607B8B', label='Nonstationary',linewidth=1.2,linestyle='--')
    plt.plot(x,np.concatenate((data_x[80][-48:],data_PatchTST[80]),axis=0).flatten(),color='#87CEEB', label='PatchTST',linewidth=1.2,linestyle='--')
    plt.plot(x,np.concatenate((data_x[80][-48:],data_LlaLinear[80]),axis=0).flatten(),color='#FFA500', label='LLM_Linear',linewidth=1.5)
    plt.plot(x,np.concatenate((data_x[80][-48:],data_true[80]),axis=0).flatten(),color='#EE6363', label='GroundTruth',linewidth=1.5)
    plt.legend(fontsize='small', framealpha=0.7)
    # 设置刻度朝内
    plt.tick_params(direction='in')
    # 获取当前的坐标轴对象
    ax = plt.gca()
    # 添加竖线
    ax.axvline(47)
    plt.show()
    plt.savefig('few_result.svg', bbox_inches='tight')
    
    


visual_few()

