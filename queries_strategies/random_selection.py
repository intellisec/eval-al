from .strategy import Strategy
import random
import numpy as np

class RandomSelection(Strategy):
    def __init__(self, model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device):
        super(RandomSelection, self).__init__(model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device)

    def query(self):
        random.seed(self.init_set_seed)
        arg = list(range(len(self.subset)))
        random.shuffle(arg)
        arg = np.array(arg)
        return arg
    