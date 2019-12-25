import torch
from torch import nn
from torch.nn.init import xavier_normal_
import numpy as np


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, cfg):
        super(TuckER, self).__init__()

        # self.E = torch.nn.Embedding(len(d.entities), d1)  # , padding_idx=0)
        # self.R = torch.nn.Embedding(len(d.relations), d2) #, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(cfg.input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(cfg.hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(cfg.hidden_dropout2)

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        # math:`C` from an expected input of size math:`(N, C, L)` or
        # math:`L` from input of size :math:`(N, L)`

    def evaluate(self, e1, r, es):
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        W_mat = torch.mm(r, self.W.view(r.size(1), -1))  # torch.mm() 矩阵相乘
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)  # torch.mm() batch矩阵相乘
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, es.transpose(1, 0))
        pred = torch.sigmoid(x)

        return pred

    def forward(self, e1, r, e2p, e2n):
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x_p = (x*e2p).sum(dim=1)
        x_n = (x*e2n).sum(dim=1)
        pred_p = torch.sigmoid(x_p)
        pred_n = torch.sigmoid(x_n)
        return pred_p, pred_n
