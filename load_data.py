import json

class Data:

    def __init__(self, data_dir="data/FB15k-237/", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.strip().split('\t') for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

class DataText():

    def __init__(self, data_dir="data/ATOMIC/", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.textdata = None # = Etextdata + Rtextdata; list,=get_index()
        self.Etextdata = None
        self.Rtextdata = None
        self.entities = self.get_entities(self.data) # list[string]
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split('\t') for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    # def get_index(self, data, maxlength): #return [triples, sentences, ids],designed for  [triples, sentences, words]
    #     textdata = []
    #     for i in data:
    #         for j in i:
    #             while(len(j)<maxlength):
    #                 j.append(0)
    #             if(len(j)>maxlength):
    #                 raise ("sentence length error")
    #             textdata.append(j)
    #     return textdata

    def get_index(self, data, maxlength):#return [sentences, ids],designed for  [sentences, words];
    # it is padding operation
        textdata = []
        for i in data:
            while(len(i)<maxlength):
                i.append(0)
            if(len(i)>maxlength):
                i=i[0:maxlength] #cut off too long sents
            textdata.append(i)
        return textdata

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities


def read_vocab(vocab_dir = "vocab/vocab.txt"):
    vocab = []
    with open(vocab_dir, 'r') as f_vocab:
        for i in f_vocab:
            vocab.append(i)
    return vocab


def read_json(path = 'config/config.json'):
    with open(path, 'r') as f:
        dict = json.load(f)
    return dict