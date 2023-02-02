import numpy as np
import scipy.sparse as sp

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from utils import normt_spm, spm_to_tensor


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.4)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj_set, att, residual=True):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        support = torch.mm(inputs, self.w) + self.b
        outputs = None
        for i, adj in enumerate(adj_set):
            if residual == False:
                y = (support - torch.mm(adj, support)) * att[i]
            else:
                y = torch.mm(adj, support) * att[i]
            if outputs is None:
                outputs = y
            else:
                outputs = outputs + y

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs

class CoLa_DGP(nn.Module):

    def __init__(self, n, edges_set, in_channels, out_channels, hidden_layers, lam):
        super().__init__()

        self.n = n
        self.d = len(edges_set)

        self.a_adj_set = []
        self.r_adj_set = []

        for edges in edges_set:
            edges = np.array(edges)
            adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                shape=(n, n), dtype='float32')
            a_adj = spm_to_tensor(normt_spm(adj, method='in')).cuda()
            r_adj = spm_to_tensor(normt_spm(adj.transpose(), method='in')).cuda()
            self.a_adj_set.append(a_adj)
            self.r_adj_set.append(r_adj)

        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        self.a_att = nn.Parameter(torch.ones(self.d))
        self.r_att = nn.Parameter(torch.ones(self.d))

        self.h = [in_channels]

        attention_weights = []
        weight = nn.Linear(in_channels, in_channels, bias=False)
        nn.init.normal_(weight.weight, std=0.01)
        self.add_module('attention-weight-0', weight)
        attention_weights.append(weight)

        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)
            self.h.append(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv1-{}'.format(i), conv)
            layers.append(conv)

            weight = nn.Linear(c, c, bias=False)
            nn.init.normal_(weight.weight, std=0.01)
            self.add_module('attention-weight-{}'.format(i), weight)
            attention_weights.append(weight)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv1-last', conv)
        layers.append(conv)

        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv2-{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv2-last', conv)
        layers.append(conv)

        self.layers = layers
        self.attention_weights = attention_weights
        self.sig = nn.Sigmoid()

        self.lam = lam

    def forward(self, x):
        conv = self.layers

        adj_set = self.a_adj_set
        att = self.a_att
        att = F.softmax(att, dim=0)
        adj0_a = None
        for i, adj in enumerate(adj_set):
            if adj0_a is None:
                adj0_a = adj * att[i]
            else:
                adj0_a += adj * att[i]
        adj0_a = adj0_a.detach()

        x1, x2 = x, x

        attW = self.sig(torch.sum(self.attention_weights[0](torch.mm(adj0_a, x1)) * x1 / math.sqrt(self.h[0]), dim=1, keepdim=True))
        y1 = conv[0](x1, adj_set, att)
        y2 = conv[2](x2, adj_set, att, residual=False)
        x1 = y1 + self.lam * (1 - attW) * y2
        x2 = y2 + attW * y1

        adj_set = self.r_adj_set
        att = self.r_att
        att = F.softmax(att, dim=0)
        adj0_r = None
        for i, adj in enumerate(adj_set):
            if adj0_r is None:
                adj0_r = adj * att[i]
            else:
                adj0_r += adj * att[i]
        adj0_r = adj0_r.detach()

        attW = self.sig(torch.sum(self.attention_weights[1](torch.mm(adj0_r, x1)) * x1 / math.sqrt(self.h[1]), dim=1, keepdim=True))
        y1 = conv[1](x1, adj_set, att)
        y2 = conv[3](x2, adj_set, att, residual=False)
        x1 = y1 + self.lam * (1 - attW) * y2
        x2 = y2 + attW * y1

        return F.normalize(x1)

