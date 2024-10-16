import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from utils.tools import series_decomp
def get_mse(y_pred,y_true):
    MSE = torch.mean((y_pred - y_true)**2)
    MSE = np.round(MSE.item(),4)
    return MSE

def visual_trend(setting, name='./pic/test.pdf'):
    folder_path = './results_trend/' + setting + '/'
    """
    Trend results visualization
    """
    data_x = np.load(folder_path+'x.npy')
    data_true = np.load(folder_path+'true.npy')
    data_pred = np.load(folder_path+'pred.npy')
    
    data_true = torch.Tensor(data_true[220]).reshape(1,720,1)
    data_pred = torch.Tensor(data_pred[220]).reshape(1,720,1)
    trend_decom = series_decomp(721)
    _, data_true_trend = trend_decom(data_true)
    _, data_pred_trend = trend_decom(data_pred)
    all_mse = get_mse(data_pred,data_true)
    trend_mse = get_mse(data_pred_trend,data_true_trend)
    x = np.arange(0, 1232)
    x1 = np.arange(512,1232)
    plt.figure(figsize=(12,2))
    plt.xlim(x[0], x[-1])
    # 获取当前的坐标轴对象
    ax = plt.gca()

    # 获取x轴的范围
    x_range = ax.get_xlim()

    # 计算中间位置
    midpoint = (x_range[0] + x_range[1])*(512/1232)

    # 添加竖线
    ax.axvline(512)
    plt.plot(x,np.concatenate((data_x[220].flatten(),data_true[0].flatten())),color='#666666', label='GroundTruth',linewidth=1)
    plt.plot(x1,data_pred[0].flatten(),label='Prediction',color='#FF8C00', linewidth=1)
    plt.plot(x1,data_true_trend.view(-1).detach().cpu().numpy(),label='Trend',color='#666666', linewidth=1,linestyle='--')

    text1 = plt.text(x=33,#文本x轴坐标 
         y=2.2, #文本y轴坐标
         s='———', #文本内容
         
         fontdict=dict(fontsize=8, color='#FF8C00',family='monospace',),#字体属性字典
         
         #添加文字背景色
         bbox={'facecolor': '#FFFFFF', #填充色
              'edgecolor':'#DCDCDC',#外框色
               'alpha': 0.5, #框透明度
               'pad': 2,#本文与框周围距离 
              }
         
        )
    text2 = plt.text(x=65,#文本x轴坐标 
         y=2.2, #文本y轴坐标
         s = str(setting)[8:]+','+'MSE:'+str(all_mse)+', '+'Trend MSE:'+str(trend_mse),
        #  s='Ours, MSE:0.1263, Trend MSE:0.0493', #文本内容
         
         fontdict=dict(fontsize=9, color='#616161',family='monospace',),#字体属性字典
         
         #添加文字背景色
         bbox={'facecolor': '#FFFFFF', #填充色
              'edgecolor':'#DCDCDC',#外框色
               'alpha': 0.5, #框透明度
               'pad': 2,#本文与框周围距离 
              }
         
        )
    # 设置x轴刻度为空
    plt.xticks([])
    # 在指定位置添加自定义文本
    # plt.text(2,1, "--ours", fontsize=5, color='red',)
    

    # plt.title('--ours','left')
    # if preds is not None:
    #     plt.plot(preds, label='Prediction', linewidth=2)
    # plt.legend()
    plt.show()
    plt.savefig(str(setting)+'.jpg', bbox_inches='tight')
visual_trend('weather_TimesNet')