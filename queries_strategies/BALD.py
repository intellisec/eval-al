import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
#custom
from .strategy import Strategy
from data.sampler import SubsetSequentialSampler

class BALD(Strategy):
    def __init__(self, model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device):
        super(BALD, self).__init__(model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device)

    def query(self):
        unlabeled_loader = DataLoader(self.data_unlabeled, batch_size=self.BATCH, 
                                    sampler=SubsetSequentialSampler(self.subset), 
                                    pin_memory=True)
        n_uPts = len(self.subset)
    
        # Heuristic:m G_X - F_X
        score_ALL = np.zeros(shape=(n_uPts, self.NO_CLASSES))
        all_entropy_dropout = np.zeros(shape=(n_uPts))

        for d in tqdm(
            range(self.cfgs.BALD.DROPOUT_ITER),
            desc = "Dropout Iterations",
        ):
            probs = self.get_predict_prob(unlabeled_loader).cpu().numpy()
            score_ALL += probs

            # computing F_X
            dropout_score_log = np.log2(
                probs + 1e-6
            )# add 1e-6 to avoid log(0)
            Entropy_Compute = -np.multiply(probs, dropout_score_log)
            Entropy_per_Dropout = np.sum(Entropy_Compute, axis=1)

            all_entropy_dropout += Entropy_per_Dropout

        Avg_Pi= np.divide(score_ALL, self.cfgs.BALD.DROPOUT_ITER)
        Log_Avg_Pi = np.log2(Avg_Pi + 1e-6)
        Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
        G_X = np.sum(Entropy_Avg_Pi, axis=1)
        F_X =np.divide(all_entropy_dropout, self.cfgs.BALD.DROPOUT_ITER)
        U_X = G_X - F_X
        arg = np.argsort(U_X)
        return arg

    def get_predict_prob(self, unlabeled_loader):
        self.model['backbone'].eval()
        with torch.cuda.device(self.device):
            predic_probs = torch.tensor([]).cuda()

        with torch.no_grad():
            for inputs, _, _ in unlabeled_loader:
                with torch.cuda.device(self.device):
                    inputs = inputs.cuda()
                predict, _, _ = self.model['backbone'](inputs)
                prob = F.softmax(predict, dim=1)
                predic_probs = torch.cat((predic_probs, prob), 0)
        return predic_probs