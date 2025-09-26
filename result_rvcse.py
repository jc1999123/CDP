from tkinter import Y
from transformer_vae_cell_rvcse_new import VAE
import numpy as np

import matplotlib.pyplot as plt
import torch

import random
from torch import nn
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from data import dataread, dataread_small,dataread_rvcse,dataread_sciplex
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from data_test import dataread_sciplex_test ,dataread_rvcse_other,dataread_rvcse_in
class dataload:
    def __init__(self,
                 data_inputx,
                 data_inputy,
                 sizetrain=10,
                 sizetest=10,
                 ):
        data = np.column_stack([data_inputx, data_inputy])
        #  print(data_inputx.shape)
        #  exit()
        indices = np.random.randint(0, high=data_inputx.shape[0], size=sizetrain)
        indicesy = np.random.randint(0, high=data_inputx.shape[0], size=sizetest)
        #  sampled_data = data[indices]
        #  sampled_data = data[indicesy]

        self.train_x = data_inputx[indices, :]
        self.train_y = data_inputy[indices]
        self.test_x = data_inputx[indicesy, :]
        self.test_y = data_inputy[indicesy]

    #  print(self.train_x.shape)
    #  exit()

    #  self.y = sampled_data

    # 提取数据

    def training_set(self):
        return torch.Tensor(self.train_x), torch.Tensor(self.train_y)

    def test_set(self):
        return torch.Tensor(self.test_x), torch.Tensor(self.test_y)

    def predict(self, new_model):
        y_true, target = self.test_set()
        y_pred, type, input, mu, log_var = new_model(y_true)

        yy = y_pred.data.numpy()
        xx = y_true.data.numpy()
        # corr = np.corrcoef(xx, yy)

        correlations = []
        for i in range(xx.shape[0]):
            correlations.append(np.corrcoef(xx[i, :], yy[i, :]))
        print(xx.shape)

        # 计算总体相关性。
        total_correlation = sum(correlations) / 5

        xx = xx.flatten()
        yy = yy.flatten()

        corr = np.corrcoef(xx, yy)

        mse = np.mean((xx - yy) ** 2)
        # print(xx,yy)
        print('pearson', corr, "totalpearson", total_correlation, 'mse', mse)
        exit()
        return corr, mse


class Testdataload:
    def __init__(self,
                 data_inputx,
                 data_inputy,
                 sizetrain=100,
                 sizetest=100,
                 ):
        data = np.column_stack([data_inputx, data_inputy])
        indices = np.random.randint(0, data_inputx.shape[0], size=sizetrain)
        indicesy = np.random.randint(0, data_inputx.shape[0], size=sizetest)
        #  sampled_data = data[indices]
        #  sampled_data = data[indicesy]

        self.train_x = data_inputx[indices, :]
        self.train_y = data_inputy[indices]
        self.test_x = data_inputx[indicesy, :]
        self.test_y = data_inputy[indicesy]

    #  self.y = sampled_data

    # 提取数据

    def training_set(self):
        return torch.Tensor(self.train_x), torch.Tensor(self.train_y)

    def test_set(self):
        return torch.Tensor(self.test_x), torch.Tensor(self.test_y)

    def predict(self, new_model):
        y_true, target = self.test_set()
        y_pred, type, input, mu, log_var = new_model(y_true)

        yy = y_pred.data.numpy()
        xx = y_true.data.numpy()
        target = target.data.numpy().flatten()
        type = type.data.numpy().flatten()

        correlations = []
        for i in range(xx.shape[0]):
            correlations.append(np.corrcoef(xx[i, :], yy[i, :]))
        print(xx.shape)

        # 计算总体相关性。
        total_correlation = sum(correlations) / xx.shape[0]

        xx = xx.flatten()
        yy = yy.flatten()

        # corr = np.corrcoef(xx, yy)

        # mse = np.mean((xx - yy) ** 2)
        # print(xx,yy)
        # print('pearson',corr,"totalpearson",total_correlation,'mse',mse)
        corr = np.corrcoef(xx, yy)

        mse = np.mean((xx - yy) ** 2)
        # print(xx,yy)
        auc = roc_auc_score(target, type)
        r2 = r2_score(xx,  yy)
        print('pearson', corr, "totalpearson", total_correlation, 'r2',r2,'mse', mse, 'auc', auc)
        # exit()

        return corr, mse, auc



def predict(y_true, y_pred):
        # y_true, target = self.test_set()
        # y_pred, type, input, mu, log_var = new_model(y_true)

        yy = y_pred.data.numpy()
        xx = y_true.data.numpy()
        # target = target.data.numpy().flatten()
        # type = type.data.numpy().flatten()

        correlations = []
        for i in range(xx.shape[0]):
            correlations.append(np.corrcoef(xx[i, :], yy[i, :]))
        print(xx.shape)

        # 计算总体相关性。
        total_correlation = sum(correlations) / xx.shape[0]
        r2total=[]
        r2 = r2_score(xx,  yy)
        for i in range(xx.shape[0]):
            r2total.append(r2_score(xx[i, :], yy[i, :]))
        print(xx.shape)
        for i in range(len(r2total)):
            if r2total[i]<0:
                r2total[i] =0

        total_r2 = sum(r2total) / xx.shape[0]


        xx = xx.flatten()
        yy = yy.flatten()

        # corr = np.corrcoef(xx, yy)

        # mse = np.mean((xx - yy) ** 2)
        # print(xx,yy)
        # print('pearson',corr,"totalpearson",total_correlation,'mse',mse)
        corr = np.corrcoef(xx, yy)

        mse = np.mean((xx - yy) ** 2)
        # print(xx,yy)
        # auc = roc_auc_score(target, type)

        r2 = r2_score(xx,  yy)

        print('pearson', corr[0,1], "totalpearson", total_correlation, 'r2',r2,'mse', mse,'r2total',total_r2)
        # exit()

        return corr, mse 



# file = 'Integrated_.h5ad'
# X_train, X_test, y_train, y_test = dataread(file)

# file = 'sciplex_othermodel.h5ad'
# X_train, X_test =dataread_sciplex_other(file)
# file = 'sciplex_othermodel.h5ad'
file = 'rvcse_221021.h5ad'
x, y = dataread_rvcse_other(file)
# x, y = dataread_rvcse_in(file)
# exit()
# X_train =x[:-1,:]
# y_train =y[:-1,:]
# X_pre = x[1:,:]
# y_pre = y[1:,:]
length_count =100
X_train =x[0:0+length_count,:]
y_train =y[0:0+length_count,:]
X_pre = x[1:1+length_count,:]
y_pre = y[1:1+length_count,:]
print(X_train.shape,y_train.shape,X_pre.shape, y_pre.shape)

model = VAE()
model_path ="savemodel/bas_bru_train/model_Bas_Bas+PN_layer1.15.pt"
model.load_state_dict(torch.load(model_path))
model.eval()
X_train = torch.Tensor(X_train)
X_pre = torch.Tensor(X_pre)
latent_dim = 128
###train
mu_train, log_var_train = model.encode( X_train)
z_train = model.reparameterize(mu_train, log_var_train)
z_causal_train   = torch.split(z_train, int(latent_dim / 8) , dim=1)[0]

###pre
mu_pre, log_var_pre = model.encode( X_pre)
z_pre = model.reparameterize(mu_pre, log_var_pre)
z_causal_pre   = torch.split(z_pre, int(latent_dim / 8) , dim=1)[0]
print(z_train.shape ,z_causal_train.shape,z_pre.shape,z_causal_pre.shape ,"shape")
# z_pre,z_cau_pre,z_canshu = model.hidden_pre( torch.Tensor(X_pre))
z_train2pre =z_train
z_train2pre[:, :16] = z_pre[:, :16]
print(z_pre[:, :16].shape)

out =model.decode(z_train2pre)

corr, mse =predict(X_pre,out)