import argparse
import os
import torch
import random
import pandas as pd
import numpy as np
import math
import sys
from pathlib import Path
from models.Lla_Arima import Model
import matplotlib as mpl
import matplotlib.pyplot as plt
from darts.datasets import AirPassengersDataset, WineDataset,MonthlyMilkDataset
from sklearn.preprocessing import MinMaxScaler
from utils.tools import visual
import darts
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='Not implemented')
parser.add_argument('--embed', type=str, default='timeF',
                    help='Not implemented')

# data loader
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='S',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--sample', type=float, default=0.01, help='Sampling percentage, the inference time of ARIMA and SARIMA is too long, you might sample 0.01')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
                    
# forecasting task
parser.add_argument('--seq_len', type=int, default=512, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length') # Just for reusing data loader
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--d_ff', type=int, default=896, help='dimension of fcn')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--percent', type=int, default=100)

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='Not implemented')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--batch_size', type=int, default=100, help='batch size of train input data')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--AutoCon_wnorm', type=str, default='ReVIN', help='Window Normalization Techniques: {ReVIN | Mean | LastVal | Decomp}')
# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
mpl.rcParams.update(mpl.rcParamsDefault)



class ExpStat():
    def __init__(self,dataname) -> None:
        path = './data/memorization/'
        self.dataname = dataname
        dataname = os.path.join(path,self.dataname)
        self.data = pd.read_csv(dataname)  # 假设数据集保存为CSV文件
        # print(self.data)
        scaler = MinMaxScaler()
        # 拟合数据并进行转换
        self.data = scaler.fit_transform(np.array(self.data[args.target]).reshape(len(self.data['OT']),1))
        # date = pd.to_datetime(self.data['date'])  # 将日期列转换为datetime类型
        # date = self.data['date']
        print(self.data)
        MyDevice = torch.device('cuda:0')
        args.seq_len = math.ceil(0.8*len(self.data))
        args.pred_len = len(self.data) - args.seq_len
        
        self.x_enc = torch.Tensor(self.data[:args.seq_len]).reshape((1,args.seq_len,1))

    def forward(self):
        model = Model(args).float()
        forecast = model(self.x_enc)
        forecast = forecast.flatten().detach().cpu().numpy()
        basic = np.array(self.data[0:args.seq_len]).flatten()
        true = np.array(self.data[args.seq_len:args.seq_len+args.pred_len]).flatten()
        np.save(r'./results_stat/'+self.dataname[:-3]+'forecast.npy',forecast)       
        np.save(r'./results_stat/'+self.dataname[:-3]+'basic.npy',basic)
        np.save(r'./results_stat/'+self.dataname[:-3]+'true.npy',true)
        
        # 可视化预测结果color='RoyalBlue'color='DarkOrange'
        plt.figure()
        plt.plot(np.concatenate([basic, true]), label='GroundTruth')
        plt.plot(np.concatenate([basic, forecast]), label='Prediction')
        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(r'./results_stat/'+self.dataname[:-3]+'svg')
        plt.show()





folder_path = "./data/memorization/"  # 设置文件夹路径
p = Path(folder_path)
filenames = [file.name for file in p.iterdir() if file.is_file()]

for filename in filenames[0:]:
    try:
        exp = ExpStat(filename)
        exp.forward()
    except:
        pass
        
    
        