import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
import math
import os
import random
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import date
import argparse
from progressbar import *


class Gaussian(nn.Module):

    def __init__(self, hidden_size, output_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)

    def forward(self, h):  # h为神经网络隐藏层输出 (batch, hidden_size)
        _, hidden_size = h.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        mu_t = self.mu_layer(h)
        return mu_t, sigma_t  # (batch, output_size)


class NegativeBinomial(nn.Module):

    def __init__(self, input_size, output_size):
        '''
        Negative Binomial Supports Positive Count Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(NegativeBinomial, self).__init__()
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)

    def forward(self, h):  # h为神经网络隐藏层输出 (batch, hidden_size)
        _, hidden_size = h.size()
        alpha_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        mu_t = torch.log(1 + torch.exp(self.mu_layer(h)))
        return mu_t, alpha_t  # (batch, output_size)


def gaussian_sample(mu, sigma):
    '''
    Gaussian Sample
    Args:
    ytrue (array like)
    mu (array like) # (num_ts, 1)
    sigma (array like): standard deviation # (num_ts, 1)
    gaussian maximum likelihood using log
        l_{G} (z|mu, sigma) = (2 * pi * sigma^2)^(-0.5) * exp(- (z - mu)^2 / (2 * sigma^2))
    '''
    # likelihood = (2 * np.pi * sigma ** 2) ** (-0.5) * \
    #         torch.exp((- (ytrue - mu) ** 2) / (2 * sigma ** 2))
    # return likelihood
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    ypred = gaussian.sample()
    return ypred  # (num_ts, 1)


def negative_binomial_sample(mu, alpha):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)
    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))
    minimize loss = - log l_{nb}
    Note: torch.lgamma: log Gamma function
    '''
    var = mu + mu * mu * alpha
    ypred = mu + torch.randn() * torch.sqrt(var)
    return ypred


class DeepAR(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, lr=1e-3, likelihood="g"):
        super(DeepAR, self).__init__()

        # network
        self.input_embed = nn.Linear(1, embedding_size)
        self.encoder = nn.LSTM(embedding_size + input_size, hidden_size, \
                               num_layers, bias=True, batch_first=True)
        if likelihood == "g":
            self.likelihood_layer = Gaussian(hidden_size, 1)
        elif likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(hidden_size, 1)
        self.likelihood = likelihood

    def forward(self, X, y, Xf):
        '''
        Args:
        num_time_series = batch_size
        X (array like): shape (num_time_series, num_obs_to_train, num_features)
        y (array like): shape (num_time_series, num_obs_to_train)
        Xf (array like): shape (num_time_series, seq_len, num_features)
        Return:
        mu (array like): shape (num_time_series, num_obs_to_train + seq_len)
        sigma (array like): shape (num_time_series, num_obs_to_train + seq_len)
        '''
        if isinstance(X, type(np.empty(2))):  # 转换为tensor
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            Xf = torch.from_numpy(Xf).float()
        num_ts, num_obs_to_train, _ = X.size()
        _, seq_len, num_features = Xf.size()
        ynext = None
        ypred = []
        mus = []
        sigmas = []
        h, c = None, None
        # 遍历所有时间点
        for s in range(num_obs_to_train + seq_len):  # num_obs_to_train为历史序列长度，seq_len为预测长度
            if s < num_obs_to_train:  # Encoder，ynext为真实值
                if s == 0:
                    ynext = torch.zeros((num_ts, 1)).to(device)
                else:
                    ynext = y[:, s - 1].view(-1, 1)  # (num_ts,1) # 取上一时刻的真实值
                yembed = self.input_embed(ynext).view(num_ts, -1)  # (num_ts,embedding_size)
                x = X[:, s, :].view(num_ts, -1)  # (num_ts,num_features)
            else:  # Decoder，ynext为预测值
                if s == num_obs_to_train: ynext = y[:, s - 1].view(-1, 1)  # (num_ts,1) # 预测的第一个时间点取上一时刻的真实值
                yembed = self.input_embed(ynext).view(num_ts, -1)  # (num_ts,embedding_size)
                x = Xf[:, s - num_obs_to_train, :].view(num_ts, -1)  # (num_ts,num_features)
            x = torch.cat([x, yembed], dim=1)  # (num_ts, num_features + embedding)
            inp = x.unsqueeze(1)  # (num_ts,1, num_features + embedding)

            if h is None and c is None:
                out, (h, c) = self.encoder(inp)  # h size (num_layers, num_ts, hidden_size)
            else:
                out, (h, c) = self.encoder(inp, (h, c))
            hs = h[-1, :, :]  # (num_ts, hidden_size)
            hs = F.relu(hs)  # (num_ts, hidden_size)
            mu, sigma = self.likelihood_layer(hs)  # (num_ts, 1)
            mus.append(mu.view(-1, 1))
            sigmas.append(sigma.view(-1, 1))
            if self.likelihood == "g":
                ynext = gaussian_sample(mu, sigma)  # (num_ts, 1)
            elif self.likelihood == "nb":
                alpha_t = sigma
                mu_t = mu
                ynext = negative_binomial_sample(mu_t, alpha_t)  # (num_ts, 1)
            # if without true value, use prediction
            if s >= num_obs_to_train and s < num_obs_to_train + seq_len:  # 在预测区间内
                ypred.append(ynext)
        ypred = torch.cat(ypred, dim=1).view(num_ts, -1)  # (num_ts, seq_len)
        mu = torch.cat(mus, dim=1).view(num_ts, -1)  # (num_ts, num_obs_to_train + seq_len)
        sigma = torch.cat(sigmas, dim=1).view(num_ts, -1)  # (num_ts, num_obs_to_train + seq_len)
        return ypred, mu, sigma

