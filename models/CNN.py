import torch
from torch import nn
from torch.nn.init import xavier_normal_
from torch.nn import Conv1d, ReLU, MaxPool1d, Sequential
import numpy as np
from .Tucker_pn import TuckER

class CNNTuckER(nn.Module):
    """Text Encoding Model CNN"""

    def __init__(self, d, es_idx, ent_vec_dim, rel_vec_dim, cfg, max_length, Evocab=40990, Rvocab=13):
        super(CNNTuckER, self).__init__()
        self.Eembed = nn.Embedding(Evocab, cfg.hSize, padding_idx=0)
        self.Rembed = nn.Embedding(Rvocab, cfg.hSize, padding_idx=0)
        self.es_idx = es_idx
        self.Evocab = Evocab
        self.ecnn = Sequential(Conv1d(in_channels=cfg.hSize, out_channels=ent_vec_dim, kernel_size=cfg.window_size),
                             MaxPool1d(kernel_size=max_length-cfg.window_size+1))
        self.rcnn = Sequential(Conv1d(in_channels=cfg.hSize, out_channels=rel_vec_dim, kernel_size=1),
                             MaxPool1d(kernel_size=1))

        self.tucker = TuckER(d, ent_vec_dim, rel_vec_dim, cfg)
        self.loss = torch.nn.BCELoss()

    def cal_es_emb(self):
        es_tmp = self.Eembed(self.es_idx[0])
        es_tmp = torch.unsqueeze(es_tmp, 0)
        es_tmp = es_tmp.permute(0, 2, 1)
        es_encoded = self.ecnn(es_tmp)
        length = self.es_idx.size(0)
        for i in range(1, length, int(length / 10)):
            es_tmp = self.Eembed(self.es_idx[i:min(i + int(length / 10), length)])
            es_tmp = es_tmp.permute(0, 2, 1)
            es_tmp = self.ecnn(es_tmp)

            es_encoded = torch.cat((es_encoded, es_tmp), 0)
        es_encoded = es_encoded.reshape(-1, es_encoded.size(1))
        return es_encoded

    def evaluate(self, e1, r):

        e1 = self.Eembed(e1)
        r = self.Rembed(r)

        e1 = e1.permute(0, 2, 1)
        r = r.permute(0, 2, 1)

        e1_encoded = self.ecnn(e1)
        r_encoded = self.rcnn(r)

        e1_encoded = e1_encoded.reshape(-1, e1_encoded.size(1))
        r_encoded = r_encoded.reshape(-1, r_encoded.size(1))

        return self.tucker.evaluate(e1_encoded, r_encoded, self.cal_es_emb())

    def evaluate_top(self, e1, r, e2):
        e1 = self.Eembed(e1)
        r = self.Rembed(r)
        e2 = self.Eembed(e2)

        e1 = e1.permute(0, 2, 1)
        r = r.permute(0, 2, 1)
        e2 = e2.permute(0, 2, 1)
        #print('e size = ' + str(e.size()))
        e1_encoded = self.ecnn(e1)
        r_encoded = self.rcnn(r)
        e2_encoded = self.ecnn(e2)


        e1_encoded = e1_encoded.reshape(-1, e1_encoded.size(1))
        r_encoded = r_encoded.reshape(-1, r_encoded.size(1))
        e2_encoded = e2_encoded.reshape(-1, e2_encoded.size(1))

        return self.tucker.evaluate_top(e1_encoded, r_encoded, e2_encoded)


    def forward(self, e1, r, e2p, e2n):
        e1 = self.Eembed(e1)
        r = self.Rembed(r)
        e2p = self.Eembed(e2p)
        e2n = self.Eembed(e2n)

        e1 = e1.permute(0, 2, 1)
        r = r.permute(0, 2, 1)
        e2p = e2p.permute(0, 2, 1)
        e2n = e2n.permute(0, 2, 1)
        #print('e size = ' + str(e.size()))
        e1_encoded = self.ecnn(e1)
        r_encoded = self.rcnn(r)
        e2p_encoded = self.ecnn(e2p)
        e2n_encoded = self.ecnn(e2n)

        e1_encoded = e1_encoded.reshape(-1, e1_encoded.size(1))
        r_encoded = r_encoded.reshape(-1, r_encoded.size(1))
        e2p_encoded = e2p_encoded.reshape(-1, e2p_encoded.size(1))
        e2n_encoded = e2n_encoded.reshape(-1, e2n_encoded.size(1))


        return self.tucker(e1_encoded, r_encoded, e2p_encoded, e2n_encoded)



