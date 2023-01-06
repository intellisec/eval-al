#!/usr/bin/env python
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from scipy.sparse import vstack
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.text import TfidfTransformer

class ImdbDataset(Dataset) :
    url         = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    base_folder = 'aclImdb'
    filename    = 'labeledBow.feat'
    n_classes   = 2
    n_features  = 89527

    def __init__(self, root, split = 'train', download = False, samples_per_class = 25000) :
        super().__init__()
        self.root = Path(root)
        self.path = self.root / self.base_folder / split

        # Download dataset
        if download :
            self.download()
        if not self.path.exists() :
            raise RuntimeError(f'dataset not found in location {self.root}')
        
        # Load dataset as TF-IDF-weighted bag-of-words features, and binary sentiment labels
        self.data, self.targets = self.load_data(samples_per_class)
    
    def __len__(self) :
        return self.data.shape[0]

    def __getitem__(self, idx) :
        sample = self.data[idx].toarray().squeeze()
        sample = torch.tensor(sample, dtype = torch.float32)
        target = self.targets[idx]
        return sample, target

    def download(self) :
        if not self.path.exists() :
            download_and_extract_archive(self.url, self.root)
    
    def load_data(self, samples_per_class) :
        # Load bag-of-words features and transform targets to binary labels
        data, targets = load_svmlight_file(str(self.path / self.filename), n_features = self.n_features)
        targets = (targets > 5).astype(int)

        # Only take specific number of samples per class
        data_by_class = []
        targets_by_class = []
        for lbl in range(2) :
            data_by_class.append(data[targets == lbl][:samples_per_class])
            targets_by_class.append(targets[targets == lbl][:samples_per_class])
        data = vstack(data_by_class)
        targets = np.hstack(targets_by_class)

        # Calculate TF-IDF
        tfidf = TfidfTransformer()
        data = tfidf.fit_transform(data)
        return data, targets
