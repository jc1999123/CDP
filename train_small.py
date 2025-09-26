from tkinter import Y
from transformer_vae_cell_rvcse import VAE
import numpy as np

import matplotlib.pyplot as plt
import torch

import random
from torch import nn
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from data import dataread, dataread_small,dataread_rvcse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score

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


# file = 'Integrated_.h5ad'
# X_train, X_test, y_train, y_test = dataread(file)

file = 'rvcse_221021.h5ad'
X_train, X_test, y_train, y_test =dataread_rvcse(file)
# X_train =X_train.T
# X_test =X_test.T
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)

# print(y_test.shape)
# exit()
TRAIN_SIZE = 10000
TEST_SIZE = 1000
# #SINE_TRAIN = [dataload(X_train, y_train) for _ in range(TRAIN_SIZE)]
print("finish")
# print(SINE_TRAIN)
# wave = random.sample(SINE_TRAIN, 1)[0]
# print(wave)
# print('',SINE_TRAIN)
# exit()
# #SINE_TEST = [dataload(X_train, y_train) for _ in range(TEST_SIZE)]


# exit()
# SineWaveTask().plot()
# SineWaveTask().plot()
# SineWaveTask().plot()
# plt.show()


def loss_vae(recons, input, mu, log_var) -> dict:
    # recons = args[0]

    kld_weight = 0.00025  # Account for the minibatch samples from the dataset
    recons_loss = F.mse_loss(recons, input)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    loss = recons_loss + kld_weight * kld_loss
    return loss


def loss_type(y_label, y_label_pre):
    # print(y_label_pre)
    y_label_pre = y_label_pre.reshape(-1)
    y_label = y_label.reshape(-1)
    # print(y_label_pre)
    '''
    if torch.any(y_label_pre > 1).item() :
        #zero_pre =torch.zeros_like(y_label_pre)
        one_pre = torch.ones_like(y_label_pre)
        #y_label_pre =torch.where(y_label_pre <0,zero_pre,y_label_pre)
        y_label_pre =torch.where(y_label_pre >1,one_pre,y_label_pre)
    if torch.any(y_label_pre < 0).item():
        zero_pre =torch.zeros_like(y_label_pre)
        y_label_pre =torch.where(y_label_pre <0,zero_pre,y_label_pre)

        # print(y_label_pre)
    if torch.any(y_label > 1).item() :
        #zero_pre =torch.zeros_like(y_label_pre)
        one = torch.ones_like(y_label)
        #y_label_pre =torch.where(y_label_pre <0,zero_pre,y_label_pre)
        y_label =torch.where(y_label >1,one,y_label)
    if torch.any(y_label < 0).item():
        zero =torch.zeros_like(y_label)
        y_label =torch.where(y_label <0,zero,y_label)

    # print(y_label_pre)
    # exit()
    # loss_fc =torch.nn.BCELoss()
    '''
    loss_fc = torch.nn.CrossEntropyLoss()

    loss = loss_fc(y_label_pre, y_label)
    return loss


def do_base_learning(model, wave, lr_inner, n_inner):
    new_model = VAE()
    new_model.load_state_dict(model.state_dict())  # copy? looks okay
    inner_optimizer = torch.optim.SGD(new_model.parameters(), lr=lr_inner)
    # K steps of gradient descent
    for i in range(n_inner):
        y_true, target = wave.training_set()
        # x = Variable(x[:, None])
        # y_true = Variable(y_true[:, None])

        y_pred, type, input, mu, log_var = new_model(y_true)
        # print()

        # loss = ((y_pred - y_true) ** 2).mean()
        loss = loss_vae(y_pred, input, mu, log_var)
        loss_y = loss_type(target, type)
        loss = loss + 0.5 * loss_y
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
    return new_model


'''
def do_base_learning_adam(model, wave, lr_inner, n_inner, state=None):
    new_model = VAE()
    new_model.load_state_dict(model.state_dict())  # copy? looks okay
    inner_optimizer = torch.optim.SGD(new_model.parameters(), lr=lr_inner)
    if state is not None:
        inner_optimizer.load_state_dict(state)
    # K steps of gradient descent
    for i in range(n_inner):
        x, y_true, target = wave.training_set()
        # x = Variable(x[:, None])
        # y_true = Variable(y_true[:, None])

        y_pred, type, input, mu, log_var = new_model(y_true)
        # print()

        # loss = ((y_pred - y_true) ** 2).mean()
        loss = loss_function(y_pred, input, mu, log_var)

        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
    return new_model, inner_optimizer.state_dict()
'''


def do_base_eval(new_model, wave):
    y_true, target = wave.test_set()
    # x = Variable(x[:, None])
    # y_true = Variable(y_true[:, None])

    y_pred, type, input, mu, log_var = new_model(y_true)

    # loss = ((y_pred - y_true) ** 2).mean()
    loss = loss_vae(y_pred, input, mu, log_var)
    loss_y = loss_type(target, type)
    loss = loss + loss_y

    return loss.item()


def reptile_sine(model, iterations, X_test, y_test, lr_inner=0.001,
                 lr_outer=0.0001, n_inner=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_outer)

    train_metalosses = []
    test_metalosses = []

    inner_optimizer_state = None

    # Sample an epoch by shuffling all training tasks
    for t in range(iterations):

        # Sample task
        # wave = random.sample(SINE_TRAIN, 1)[0]
        wave = dataload(X_train, y_train)

        # Take k gradient steps
        new_model = do_base_learning(model, wave, lr_inner, n_inner)
        # new_model, inner_optimizer_state = do_base_learning_adam(
        #    model, wave, lr_inner, n_inner, inner_optimizer_state)

        # Eval
        train_metaloss = do_base_eval(new_model, wave)

        # Inject updates into each .grad
        for p, new_p in zip(model.parameters(), new_model.parameters()):
            if p.grad is None:
                p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.add_(p.data - new_p.data)

        # Update meta-parameters
        optimizer.step()
        optimizer.zero_grad()

        ############# Validation
        # wave = random.sample(SINE_TEST, 1)[0]
        wave = dataload(X_train, y_train)
        new_model = do_base_learning(model, wave, lr_inner, n_inner)
        test_metaloss = do_base_eval(new_model, wave)

        ############# Log
        train_metalosses.append(train_metaloss)
        test_metalosses.append(test_metaloss)

        if t % 1000 == 0:
            print('Iteration', t)
            # 'Iteration', t
            print('AvgTrainML', np.mean(train_metalosses))
            # 'AvgTrainML', np.mean(train_metalosses)
            print('AvgTestML ', np.mean(test_metalosses))

        if t % 1000 == 0:
            wave1 = Testdataload(X_test, y_test)
            for n_inner_1 in range(4):
                new_model1 = do_base_learning(model, wave1,
                                              lr_inner=0.001, n_inner=n_inner_1)
                # print(wave.)
                corr, mse, type = wave1.predict(new_model1)
                print('no train')
                model.eval()
                corr, mse, type = wave1.predict(model)
                model.train()

            model_file_path = 'savemodel/rvcse/modelnew_type3' + str(t % 10000) +'.pt'

            # 使用 torch.save 函数保存模型
            torch.save(model.state_dict(), model_file_path)
        # if t % 10000 == 0:
        #     model_file_path = 'savemodel/rvcse/model' + str(t % 10000) +'.pt'
        #
        #     # 使用 torch.save 函数保存模型
        #     torch.save(model.state_dict(), model_file_path)
        #     # 'AvgTestML ', np.mean(test_metalosses)


model = VAE()
reptile_sine(model, iterations=500000, X_test=X_test, y_test=y_test)
# print(1)
wave = Testdataload(X_test, y_test)

# print(1)

for n_inner_ in range(4):
    new_model = do_base_learning(model, wave,
                                 lr_inner=0.01, n_inner=n_inner_)
    # print(wave.)
    corr, mse, type = wave.predict(new_model)
    # print(type)


