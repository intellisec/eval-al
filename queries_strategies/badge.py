import random
import torch
import numpy as np
from copy import deepcopy
from scipy import stats
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F
#custom
from .strategy import Strategy
from data.sampler import SubsetSequentialSampler

class Badge(Strategy):
    def __init__(self, model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device):
        super(Badge, self).__init__(model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device)

    def query(self):
        unlabeled_loader = DataLoader(self.data_unlabeled, batch_size=self.BATCH, 
                                    sampler=SubsetSequentialSampler(self.subset), 
                                    pin_memory=True)

        gradEmbedding = self.get_grad_embedding(unlabeled_loader, len(self.subset)).numpy()
        print('features shape: {}'.format(gradEmbedding.shape))
        print(self.BUDGET)
        arg = self.init_centers(gradEmbedding)
        return arg

    def get_grad_embedding(self, unlabeled_loader, len_ulb):
        embDim = self.model['backbone'].get_embedding_dim()
        self.model['backbone'].eval()
        nLab = self.NO_CLASSES
        embedding = np.zeros([len_ulb, embDim*nLab])
        ind = 0
        print('embedding shape {}'.format(embedding.shape))
        with torch.no_grad():
            for x, y, idxs in unlabeled_loader:
                # print(idxs)
                with torch.cuda.device(self.device):
                    x = x.cuda()
                    y = y.cuda()
                    scores, features_batch, _ = self.model['backbone'](x)
                    features_batch = features_batch.data.cpu().numpy()
                    batchProbs = F.softmax(scores, dim=1).data.cpu().numpy()
                    maxInds = np.argmax(batchProbs, 1)
                    print('features:{}, batchProbs: {}'.format(features_batch.shape, batchProbs.shape))
                    for j in range(len(y)):
                        # print(idxs[j],ind)
                        for c in range(nLab):
                            # if j==0:
                            #     print(c, idxs)
                            # print(idxs[j],ind)
                            if c == maxInds[j]:
                                embedding[ind][embDim * c : embDim * (c+1)] = deepcopy(features_batch[j]) * (1 - batchProbs[j][c])
                            else:
                                embedding[ind][embDim * c : embDim * (c+1)] = deepcopy(features_batch[j]) * (-1 * batchProbs[j][c])
                        ind += 1
            # print(ind)
            return torch.Tensor(embedding)

    def init_centers(self, X):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        print('#Samps\tTotal Distance')
        while len(mu) < self.BUDGET:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            # if sum(D2) == 0.0: pdb.set_trace()
            assert sum(D2) != 0.0
            D2 = D2.ravel().astype(float)
            
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        others = [i for i in range(len(X)) if i not in indsAll]
        return others + indsAll