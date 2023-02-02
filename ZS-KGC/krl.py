import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import json
import scipy.sparse as sp
import numpy as np
from utils import *
from tqdm import tqdm
import math
import argparse

class KRLConv(nn.Module):
    def __init__(self, mode, in_channel, out_channel, relu, lam, adj=None):
        super(KRLConv, self).__init__()
        self.mode = mode
        self.adj = adj
        self.lam = lam
        self.dropout = nn.Dropout(p=0.1)
        self.sig = nn.Sigmoid()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.w1 = nn.Parameter(torch.randn(self.in_channel, self.out_channel))
        self.b1 = nn.Parameter(torch.randn(self.out_channel))
        self.w2 = nn.Parameter(torch.randn(self.in_channel, self.out_channel))
        self.b2 = nn.Parameter(torch.randn(self.out_channel))
        self.wm = nn.Parameter(torch.randn(self.in_channel, self.in_channel))        
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, inputs1, inputs2=None):
        if self.mode == "FFN":
            inputs1 = self.dropout(inputs1)
            outputs = torch.mm(inputs1, self.w1) + self.b1
            if self.relu is not None:
                outputs = self.relu(outputs)
            return outputs
        elif self.mode == "GCN":
            inputs1 = self.dropout(inputs1)
            outputs = torch.mm(self.adj, torch.mm(inputs1, self.w1)) + self.b1
            if self.relu is not None:
                outputs = self.relu(outputs)
            return outputs
        else:
            mu = self.sig(torch.sum(torch.mm(torch.mm(self.adj, inputs1), self.wm) * inputs1 / math.sqrt(self.in_channel), dim=1, keepdim=True))
            inputs1 = self.dropout(inputs1)
            inputs2 = self.dropout(inputs2)
            mid1 = torch.mm(inputs1, self.w1) + self.b1
            mid1 = torch.mm(self.adj, mid1)
            mid2 = torch.mm(inputs2, self.w2) + self.b2
            mid2 = mid2 - torch.mm(self.adj, mid2)
            if self.relu is not None:
                mid1 = self.relu(mid1)
                mid2 = self.relu(mid2)
            outputs1 = mid1 + self.lam * (1-mu) * mid2
            outputs2 = mid2 + mu * mid1
            
            return outputs1, outputs2

class KRLModule(nn.Module):
    def __init__(self, mode, channels, adj):
        super(KRLModule, self).__init__()
        self.mode = mode
        self.adj = adj
        layers = []
        for i in range(len(channels)-2):
            conv = KRLConv(self.mode, channels[i], channels[i+1], relu=True, lam=0.01, adj=self.adj)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)
        conv = KRLConv(self.mode, channels[len(channels)-2], channels[len(channels)-1], relu=False, lam=0.01, adj=self.adj)
        self.add_module('conv{}'.format(len(channels)-2), conv)
        layers.append(conv)
        self.layers = layers
        
    def forward(self, x):
        if self.mode == "CoLa":
            x1 = x
            x2 = x
            for conv in self.layers:
                x1, x2 = conv(x1, x2)
            return x1
        else:
            for conv in self.layers:
                x = conv(x)
            return x

def get_adj(setting, total):
    with open(f"{setting}/setting.json", "r") as file:
        settings = file.read()
        settings = json.loads(settings)
    h, t = settings['h'], settings['t']
    edges = [(h[i], t[i]) for i in range(len(h))] + [(u, u) for u in range(total)]
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(total, total), dtype='float32')
    adj = normt_spm(adj, method='in')
    adj = spm_to_tensor(adj).cuda()
    return adj

def run(model, X, w, b, visual, lr, epoch=3000):
    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model.train()
    for i in range(epoch):
        score = torch.mm(model(X), w.T) + b
        target = torch.arange(visual).cuda()
        loss = loss_fn(score[:visual, :], target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"Epoch {i+1} -- Loss: {loss.item()}")
    print(f"Loss: {loss.item()}")
    model.eval()
    z = model(X)
    return z.detach()

def KRL(setting="ImageNet21K", mode="CoLa", channels=[2048, 2048], round=1):
    visual, total, lr = 11221, 13034, 1e-2
    X = torch.load(f"{setting}/glove_embedding.pth").cuda()
    channels = [300] + channels
    std = torch.mean(torch.norm(X, dim=1)).item()
    X /= std
    if setting == "ImageNet2012":
        visual, total = 1000, 1860
    FC = torch.load(f"{setting}/FCweights.pth").cuda()
    w, b = FC[:, :-1], FC[:, -1]
    adj = get_adj(setting, total)
    model = KRLModule(mode, channels, adj).cuda()
    repr = run(model, X, w, b, visual, lr)
    torch.save(repr, f"{setting}/Reps/{mode}-{round}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', default="ImageNet21K")
    parser.add_argument('--mode', default="CoLa")
    parser.add_argument('--channels', default="2048,2048")
    parser.add_argument('--round', default=1)
    parser.add_argument('--gpu', default='0')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setting = args.setting
    mode = args.mode
    channels = args.channels.split(",")
    channels = [int(item) for item in channels]
    round = args.round
    KRL(setting, mode, channels, round)

    