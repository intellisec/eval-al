#!/usr/bin/env python
import numpy as np
import torch
from torch.utils.data import Dataset

class OpenMLDataset(Dataset) :

    def __init__(self, dataset_id = 156, train = True, train_test_split = [50000, 10000]) :
        super().__init__()
        dataset = openml.datasets.get_dataset(dataset_id)
        features, targets, categorical, feature_names = dataset.get_data(
            dataset_format = 'array',
            target = dataset.default_target_attribute
        )
        if train :
            features = features[:train_test_split[0]]
            targets  = targets[:train_test_split[0]]
        else :
            features = features[train_test_split[0]:][:train_test_split[1]]
            targets  = targets[train_test_split[0]:][:train_test_split[1]]
        self.features = torch.from_numpy(features)
        self.targets  = torch.from_numpy(targets)
    
    def __getitem__(self, idx) :
        return self.features[idx], self.targets[idx]
    
    def __len__(self) :
        return self.features.size(0)
    
    @property
    def n_features(self) :
        return self.features.size(1)
    
    @property
    def n_classes(self) :
        return self.targets.max().item() + 1
