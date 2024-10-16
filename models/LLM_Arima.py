import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models import LLM
from pmdarima.arima import auto_arima 
# from statsmodels.tsa.arima.model import ARIMA  
from utils.tools import visual
from models.Stat_models import *



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs,flag='test'):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.flag = flag
        # Decompsition Kernel Size
        kernel_size = 25
        self.input_decom = series_decomp(kernel_size)
        self.individual = configs.individual
        self.trend_decom = series_decomp(configs.pred_len+1)
        self.seasonal_decom = series_decomp(configs.pred_len + 1)
        self.ari_pred = Arima(configs).float()
        self.channels = configs.enc_in
        self.AutoCon_wnorm = configs.AutoCon_wnorm
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
        self.lla_model = LLM.Model(configs).float()    
        for param in self.lla_model.parameters():
            param.requires_grad_(False)
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x_enc):

        # x: [Batch, Input length, Channel]
        if self.AutoCon_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std = x_enc.std(dim=1, keepdim=True).detach()
            seasonal_init = (x_enc - seq_mean) / (seq_std + 1e-5)
            trend_init = seasonal_init.clone()
        elif self.AutoCon_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seasonal_init = (x_enc - seq_mean)
            trend_init = seasonal_init.clone()
        elif self.AutoCon_wnorm == 'Decomp':
            seasonal_init, trend_init = self.input_decom(x_enc)
        elif self.AutoCon_wnorm == 'LastVal':
            seq_last = x_enc[:,-1:,:].detach()
            seasonal_init = (x_enc - seq_last)
            trend_init = seasonal_init.clone()
        else:
            raise Exception(f'Not Supported Window Normalization:{self.AutoCon_wnorm}. Use {"{ReVIN | Mean | LastVal | Decomp}"}.')
        # seasonal_init = seasonal_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])

        else:
            # seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
            #                               dtype=seasonal_init.dtype).to(seasonal_init.device)

            
            seasonal_init = seasonal_init.float().cpu().numpy()
            seasonal_output = self.ari_pred(seasonal_init)
            MyDevice = torch.device('cuda:0')
            seasonal_output = torch.tensor(seasonal_output, device=MyDevice)

            _, trend_init = self.trend_decom(trend_init)

            trend_output = self.lla_model(trend_init, trend_init)
            _, trend_output = self.trend_decom(trend_output)
            # print('seasonal_output',seasonal_output)
            # print('trend_output',trend_output)

            # trend_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
            #                                         dtype=seasonal_init.dtype).to(seasonal_init.device)
            # seq_mean = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
            #                                         dtype=seasonal_init.dtype).to(seasonal_init.device)



        seasonal_output = seasonal_output.detach().cpu()
        if self.AutoCon_wnorm == 'ReVIN':
            x = (seasonal_output + trend_output)*(seq_std+1e-5) + seq_mean
        elif self.AutoCon_wnorm == 'Mean':
            x = seasonal_output + trend_output + seq_mean
        elif self.AutoCon_wnorm == 'Decomp':
            x = seasonal_output + trend_output
        elif self.AutoCon_wnorm == 'LastVal':
            x = seasonal_output + trend_output + seq_last
        else:
            raise Exception()

            
        
       
        return x # to [Batch, Output length, Channel]
