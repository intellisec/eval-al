import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
# Custom
from config import *
from models.query_models import VAE, Discriminator, GCN, GC_VAE, CVAE, CDiscriminator, FCVAE, FCCVAE
from data.sampler import SubsetSequentialSampler
from .kCenterGreedy import kCenterGreedy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import stats
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
# from influence_function import calc_influence_function
# from Influence_function import calc_influence_function, calc_influence_function_param

class Strategy:
    def __init__(self, model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device):
        self.model = model
        self.data_unlabeled = data_unlabeled
        self.subset = []
        self.labeled_set = []
        self.cycle = 0
        self.NO_CLASSES = NO_CLASSES
        self.test_loader = test_loader
        self.cfgs = cfgs
        self.device = device
        self.init_set_seed = 0

        self.BATCH = self.cfgs.BATCH
        self.BUDGET = self.cfgs.BUDGET
        self.INIT_BUDGET = self.cfgs.INIT_BUDGET

    def query(self):
        pass

    def seed_random_init_set(self, init_set_seed):
        self.init_set_seed = init_set_seed
    
    def feed_current_state(self, cycle, subset, labeled_set):
        self.subset = subset
        self.labeled_set = labeled_set
        self.cycle = cycle

