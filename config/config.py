
class config():

    def __init__(self, dict):
        self.model = dict["model"]
        self.nL = dict["nL"]
        self.nH = dict["nH"]
        self.hSize = dict["hSize"]
        self.edpt = dict["edpt"]
        self.adpt = dict["adpt"]
        self.rdpt = dict["rdpt"]
        self.odpt = dict["odpt"]
        self.pt = dict["pt"]
        self.afn = dict["afn"]
        self.init = dict["init"]
        self.vSize = dict["vSize"]
        self.window_size = dict["window_size"]
        #self.rel_embedding = dict["rel_embedding"]
        self.input_dropout = dict["input_dropout"]
        self.hidden_dropout1 = dict["hidden_dropout1"]
        self.hidden_dropout2 = dict["hidden_dropout2"]



