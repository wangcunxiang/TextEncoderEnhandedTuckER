from load_data import DataText, read_json
import time
from collections import defaultdict
import argparse
import torch.tensor
import random
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
import os

from models.CNN import CNNTuckER
from models.LSTM import LSTMTuckER

from config.config import config

hit10 = 0
hit3 = 0
hit1 = 0
MR = 15000
MRR = 0


class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False,
                 label_smoothing=0., maxlength=5):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.maxlength = maxlength
        self.Etextdata = None
        self.Rtextdata = None
        self.Evocab = ['NULL', ]  # padding_idx=0
        self.Rvocab = ['NULL', ]  # padding_idx=0
        self.max_test_hit1 = 0.
        self.max_test_hit3 = 0.
        self.max_test_hit10 = 0.
        self.max_test_MR = 999999999.
        self.max_test_MRR = 0.


    def get_vocab_emb(self, vocab, hSize, data_dir="embedding/", data_type="word_embs.txt"):
        vocab2embs = {'NULL':[0. for i in range(hSize)], }
        with open("%s%s" % (data_dir, data_type), "r") as f:
            for line in f.readlines():
                #print('line = '+str(line))
                word, emb = line.strip().split("\t")
                emb = [float(i) for i in emb.split(',')]
                vocab2embs[word] = emb
        for item in vocab:
            if item not in vocab2embs:
                print('{} is missing'.format(item))
                vocab2embs[item]=[0. for i in range(hSize)]

        embs = [ vocab2embs[word] for word in vocab ]
        return embs

    def strings_to_ids(self, data, vocab=['NULL', ]):  # padding_idx=0; designed for [sentences, words]
        if vocab == ['NULL', ]:
            tmp = []
            for sent in data:
                sent = sent.strip().split()
                tmp += sent
            vocab += sorted(list(set(tmp)))

        vocab_ = {vocab[i]: i for i in range(len(vocab))}
        data_ids = []
        for sent in data:
            sent = sent.strip().split()
            for i, word in enumerate(sent):
                if word not in vocab_:
                    print("{} is missing".format(word))
                    sent[i] = 'UNK'
            word_ids = [vocab_[word] for word in sent]
            data_ids.append(word_ids)
        return data_ids, vocab

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch_train(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        batch_ = list((t[0], t[1]) for t in batch)
        negs = np.random.randint(len(d.entities), size=len(batch))
        for idx, pair in enumerate(batch_):
            while negs[idx] in er_vocab[pair]:
                negs[idx] = random.randint(0, len(d.entities) - 1)
        negs = torch.LongTensor(negs)
        # if self.cuda:
        #     negs = negs.cuda()
        return np.array(batch), negs

    def get_batch_eval(self, er_vocab, e1_r):
        e1 = e1_r[0]
        r = e1_r[1]
        batch = [[e1,r, e2] for e2 in er_vocab[e1_r]]

        return np.array(batch)

    def get_batch_dev(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets


    def evaluate(self, model, data, fw):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))# dict [(e1, r)]->[r2a, r2b, ...,]
        test_er_vocab = self.get_er_vocab(self.get_data_idxs(data)) # dict [(e1, r)]->[r2a, r2b, ...,]
        test_er_vocab_pairs = list(test_er_vocab.keys())  # list [...,(e1,r),...]

        print("Number of test data points: %d" % len(test_data_idxs))
        #print("test_er_vocab_pairs="+str(test_er_vocab_pairs))
        for e1_r in test_er_vocab_pairs:

            e1 = e1_r[0]
            r = e1_r[1]
            data_batch = np.array([[e1, r, e2] for e2 in er_vocab[e1_r]])
            #print("data_batch="+str(data_batch))
            #data_batch = self.get_batch_eval(test_er_vocab, e1_r)

            e1_idx = torch.LongTensor(self.Etextdata[data_batch[:, 0]])
            r_idx = torch.LongTensor(self.Rtextdata[data_batch[:, 1]])
            e2_idx = torch.LongTensor(self.Etextdata[data_batch[:, 2]])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.evaluate_top(e1_idx, r_idx, e2_idx)
            #print("predictions="+str(predictions))
            top_values, top_idxs = torch.topk(predictions, k=1)

            #print("top_idxs: "+str(top_idxs))
            tail = er_vocab[e1_r][top_idxs.tolist()[0]]
            fw.write(d.entities[e1])
            fw.write('\t')
            fw.write(d.relations[r])
            fw.write('\t')
            fw.write(d.entities[tail])
            fw.write('\n')


    def train_and_eval(self):
        print("Training the {} model on {}...".format(args.model, args.dataset))
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        # data_idxs = self.get_data_idxs(d.data)
        print("Number of training data points: %d" % len(train_data_idxs))
        # print("Number of all data points: %d" % len(data_idxs))


        ########
        # data_ids, self.vocab = self.strings_to_ids(vocab=self.vocab, data=d.data)
        #print('d.entities='+str(len(d.entities)))
        entities_ids, self.Evocab = self.strings_to_ids(vocab=['NULL', ], data=d.entities)

        #print("entities_ids = " + str(entities_ids))
        relation_ids, self.Rvocab = self.strings_to_ids(vocab=['NULL', ], data=d.relations)
        print("entities_ids len=%d" % len(entities_ids))
        print("relation_ids len=%d" % len(relation_ids))
        #print('XXX = ' + str([len(i) for i in entities_ids].index(0)))
        #print('YYY = ' + str([len(i) for i in entities_ids].index(0)))
        cfg = config(dict(read_json(args.config)))
        if args.do_pretrain == 1:
            cfg.hSize = 768
            Eembs = self.get_vocab_emb(self.Evocab, cfg.hSize)
        print("read vocab ready.")

        d.Etextdata = d.get_index(entities_ids, self.maxlength)  # list, contained padding entities
        self.Etextdata = np.array(d.Etextdata)
        d.Rtextdata = d.get_index(relation_ids, 1)
        self.Rtextdata = np.array(d.Rtextdata)
        # self.textdata = np.array(d.Etextdata + d.Rtextdata)
        #self.check_textdata()

        print("text data ready")
        es_idx = torch.LongTensor(self.Etextdata)
        if self.cuda:
            es_idx = es_idx.cuda()
            print("es ready")

        if args.model == 'CNN':
            model = CNNTuckER(d=d, es_idx=es_idx, ent_vec_dim=self.ent_vec_dim, rel_vec_dim=self.rel_vec_dim,
                              cfg=cfg, max_length=self.maxlength,
                              Evocab=len(self.Evocab), Rvocab=len(self.Rvocab))
        elif args.model == 'LSTM':
            model = LSTMTuckER(d=d, es_idx=es_idx, ent_vec_dim=self.ent_vec_dim, rel_vec_dim=self.rel_vec_dim,
                              cfg=cfg, Evocab=len(self.Evocab), Rvocab=len(self.Rvocab))
        else:
            print("No Model")
            exit(0)
        print("model ready")
        if args.do_pretrain == 1:
            model.Eembed.weight.data.copy_(torch.from_numpy(np.array(Eembs)))
            print("Embedding Loaded")


        ########
        if self.cuda:
            model.cuda()
        print("Test:")
        for i in ["hits10", "hits3", "hits1", "mr", "mrr"]:
            start_test = time.time()
            checkpoint = torch.load(args.outdir+"/modelpara_{}.pkl".format(i))
            model.load_state_dict(checkpoint['state'])
            print('Hits @10: {0}'.format(checkpoint['Hits@10']))
            print('Hits @3: {0}'.format(checkpoint['Hits@3']))
            print('Hits @1: {0}'.format(checkpoint['Hits@1']))
            print('MR: {0}'.format(checkpoint['MR']))
            print('MRR: {0}'.format(checkpoint['MRR']))
            path = args.outdir+'/results_'+i+'.txt'
            fw = open(path, 'w', encoding='utf-8')
            self.evaluate(model, d.test_data, fw)
            print(time.time() - start_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                        help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--model", type=str, default="TuckER", nargs="?",
                        help="TuckER, MeanTuckER, CNNTuckER")
    parser.add_argument("--do_pretrain", type=int, default=0, nargs="?",
                        help="Whether to use pretrained embeddings")
    parser.add_argument("--config", type=str, default="config/config.json", nargs="?",
                        help="the config file path")
    parser.add_argument("--outdir", type=str, default="./outdir/model", nargs="?",
                        help="the model save file path")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                        help="Batch size.")
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
    parser.add_argument("--max_length", type=int, default=5, nargs="?",
                        help="max sequence length.")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = DataText(data_dir=data_dir, reverse=False)
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr,
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            label_smoothing=args.label_smoothing, maxlength=args.max_length
                            )
    experiment.train_and_eval()