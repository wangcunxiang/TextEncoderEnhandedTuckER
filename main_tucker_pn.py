from load_data import Data, read_json
import numpy as np
import torch
import time
import random
from collections import defaultdict
from models.Tucker_pn0 import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
from config.config import config


class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False,
                 label_smoothing=0.):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch_eval(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            #print('er_vocab[pair]='+str(er_vocab[pair]))
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def get_batch_train(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        batch_ = list((t[0],t[1]) for t in batch)
        negs = np.random.randint(len(d.entities), size=len(batch))
        for idx, pair in enumerate(batch_):
            while negs[idx] in er_vocab[pair]:
                negs[idx] = random.randint(0, len(d.entities)-1)
        negs = torch.LongTensor(negs)
        if self.cuda:
            negs = negs.cuda()
        return np.array(batch), negs

    def develop(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])
        losses = []

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
        test_er_vocab = self.get_er_vocab(self.get_data_idxs(data))
        test_er_vocab_pairs = list(test_er_vocab.keys())  # list [...,(e1,r),...]

        print("Number of data points: %d" % len(test_data_idxs))

        es_idx = torch.LongTensor([i for i,_ in enumerate(d.entities)])
        if self.cuda:
            es_idx = es_idx.cuda()

        for i in range(0, len(test_er_vocab_pairs), self.batch_size):
            data_batch, targets = self.get_batch_eval(er_vocab, test_er_vocab_pairs, i)
            e1_idx = torch.LongTensor(data_batch[:, 0])
            r_idx = torch.LongTensor(data_batch[:, 1])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
            predictions = model.evaluate(e1_idx, r_idx, es_idx)

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            targets_ = targets.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(np.isin(sort_idxs[j], np.where(targets_[j] == 1.0)[0]))[0][0]
                ranks.append(rank+1)


                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

            loss = model.loss(predictions, targets)
            losses.append(loss.item())

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
        print('loss: {0}'.format(np.mean(losses)))


    def train_and_eval(self):
        print("Training the TuckER PN on {}...".format(args.dataset))
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        print('d.entities='+str(len(d.entities)))

        cfg = config(dict(read_json(args.config)))
        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, cfg)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        #er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        for it in range(1, self.num_iterations + 1):
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(train_data_idxs)
            for j in range(0, len(train_data_idxs), self.batch_size):
                data_batch, e2n_idx= self.get_batch_train(er_vocab, train_data_idxs, j)
                opt.zero_grad()
                e1_idx = torch.LongTensor(data_batch[:, 0])
                r_idx = torch.LongTensor(data_batch[:, 1])
                e2p_idx = torch.LongTensor(data_batch[:, 2])
                targets = torch.cat((torch.ones(e2p_idx.size(0)), torch.zeros(e2n_idx.size(0))),0)
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    e2p_idx = e2p_idx.cuda()
                    e2n_idx = e2n_idx.cuda()
                    targets = targets.cuda()
                pred_p, pred_n = model.forward(e1_idx, r_idx, e2p_idx, e2n_idx)

                predication = torch.cat((pred_p, pred_n),0)

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / len(d.entities))
                loss = model.loss(predication, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print(it)
            print(time.time() - start_train)
            print('loss: {0}'.format(np.mean(losses)))
            model.eval()
            with torch.no_grad():
                print("Valid:")
                start_test = time.time()
                self.evaluate(model, d.valid_data)
                print(time.time() - start_test)
                print("Test:")
                start_test = time.time()
                self.evaluate(model, d.test_data)
                print(time.time() - start_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                        help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                        help="Batch size.")
    parser.add_argument("--config", type=str, default="config/config.json", nargs="?",
                        help="the config file path")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                        help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                        help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                        help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                        help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                        help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                        help="Amount of label smoothing.")
    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir, reverse=False)
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr,
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            label_smoothing=args.label_smoothing)
    experiment.train_and_eval()

