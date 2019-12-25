import torch
from torch import nn
from torch.nn.init import xavier_normal_
from torch.nn import LSTM
import numpy as np
from .Tucker_pn import TuckER

class LSTMTuckER(nn.Module):
    """Text Encoding Model LSTM"""

    def __init__(self, d, es_idx, ent_vec_dim, rel_vec_dim, cfg, Evocab=40990, Rvocab=13):
        super(LSTMTuckER, self).__init__()
        self.Eembed = nn.Embedding(Evocab, cfg.hSize, padding_idx=0)
        self.Rembed = nn.Embedding(Rvocab, cfg.hSize, padding_idx=0)
        self.tucker = TuckER(d, ent_vec_dim, rel_vec_dim, cfg)
        self.es_idx = es_idx
        self.elstm = LSTM(cfg.hSize, int(ent_vec_dim/2), num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        #batch_first: If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False``
        self.rlstm = LSTM(cfg.hSize, int(rel_vec_dim/2), num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.loss = torch.nn.BCELoss()

    def cal_es_emb(self):
        es_tmp = self.Eembed(self.es_idx[0])
        es_encoded, tmp = self.elstm(torch.unsqueeze(es_tmp, 0))
        es_encoded = torch.mean(es_encoded, dim=1)
        length = self.es_idx.size(0)
        for i in range(1, length, int(length / 20)):
            es_tmp = self.Eembed(self.es_idx[i:min(i + int(length / 20), length)])
            # print("i="+str(i))
            es_tmp, tmp = self.elstm(es_tmp)
            es_tmp = torch.mean(es_tmp, dim=1)
            es_encoded = torch.cat((es_encoded, es_tmp), 0)

        return es_encoded

    def evaluate(self, e, r):
        e = self.Eembed(e)
        e_encoded, tmp = self.elstm(e)
        e_encoded = torch.mean(e_encoded, dim=1)

        r = self.Rembed(r)
        r_encoded, tmp = self.rlstm(r)
        r_encoded = torch.mean(r_encoded, dim=1)

        return self.tucker.evaluate(e_encoded, r_encoded, self.cal_es_emb())

    def forward(self, e, r, e2p, e2n):

        e = self.Eembed(e)
        e_encoded, tmp = self.elstm(e)
        e_encoded = e_encoded[:, -1,:]  # use last word's output

        e2p = self.Eembed(e2p)
        e2p_encoded, tmp = self.elstm(e2p)
        e2p_encoded = torch.mean(e2p_encoded, dim=1)

        e2n = self.Eembed(e2n)
        e2n_encoded, tmp = self.elstm(e2n)
        e2n_encoded = torch.mean(e2n_encoded, dim=1)


        r = self.Rembed(r)
        r_encoded, tmp = self.rlstm(r)
        r_encoded = r_encoded[:,-1,:]#use last word's output


        return self.tucker(e_encoded, r_encoded, e2p_encoded, e2n_encoded)