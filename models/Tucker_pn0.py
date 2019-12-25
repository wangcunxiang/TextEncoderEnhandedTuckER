import numpy as np
import torch
from torch.nn.init import xavier_normal_


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, cfg):
        super(TuckER, self).__init__()

        self.Eembed = torch.nn.Embedding(len(d.entities), d1)
        self.Rembed = torch.nn.Embedding(len(d.relations), d2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(cfg.input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(cfg.hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(cfg.hidden_dropout2)

        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

    def init(self):
        xavier_normal_(self.Eembed.weight.data)
        xavier_normal_(self.Rembed.weight.data)

    def evaluate(self, e1_idx, r_idx, es_idx):
        e1 = self.Eembed(e1_idx)
        es = self.Eembed(es_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.Rembed(r_idx)
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

    def forward(self, e1_idx, r_idx, e2p_idx, e2n_idx):
        e1 = self.Eembed(e1_idx)
        e2p = self.Eembed(e2p_idx)
        e2n = self.Eembed(e2n_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.Rembed(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))  # torch.mm() 矩阵相乘
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)  # torch.mm() batch矩阵相乘
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x_p = (x*e2p).sum(dim=1)
        x_n = (x*e2n).sum(dim=1)
        pred_p = torch.sigmoid(x_p)
        pred_n = torch.sigmoid(x_n)
        return pred_p, pred_n

