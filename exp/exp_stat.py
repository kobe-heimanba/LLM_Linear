from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import time
import os
import time
import warnings
import matplotlib.pyplot as plt
from models.Stat_models import *
from models.LLM_Arima import Model
warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Naive': Naive_repeat,
            'ARIMA': Arima,
            'SARIMA': SArima,
            'GBRT': GBRT,
            'LlaArima':Model,
        }
        model = model_dict[self.args.model](self.args).float()

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        # Sample 10% 
        samples = max(int(self.args.sample * self.args.batch_size),1)

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()
        time_start = time.time()
        iter_count = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count+=1
                # batch_x = batch_x.float().to(self.device).cpu().numpy()
                # batch_y = batch_y.float().to(self.device).cpu().numpy()
                batch_x = batch_x[:samples]
                print(batch_x.shape)
                batch_x = batch_x.float().to(self.device)
                
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:samples, -self.args.pred_len:, f_dim:]
                outputs = outputs.float().to(self.device).cpu().numpy()
                batch_y = batch_y.float().to(self.device).cpu().numpy()
                batch_x = batch_x.float().to(self.device).cpu().numpy()
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x)
                if i % 20 == 0:
                    input = batch_x
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                if i  % 20 == 0:                    
                    speed = (time.time() - time_now) / iter_count
                    
                    print('\tspeed: {:.4f}s/iter; '.format(speed,))
                    iter_count = 0
                    time_now = time.time()
        cost_time = time.time() - time_start
        speed = cost_time/len(test_data)
        print("cost time: {}".format(cost_time))
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # print('preds',preds,preds.shape)
        # print('trues',trues,trues.shape)
        
        mae, mse, rmse, smape, mape, mrae, mspe, rse, corr = metric(preds, trues)
        corr = []
        print('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}, cost:{}, speed:{}'.format(mse, mae, rse, corr,cost_time,speed))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return