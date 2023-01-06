#!/usr/bin/env python
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

class NoisyCIFAR10(CIFAR10) :

    def __init__(self, *args, transform = T.ToTensor(), noise_std = 0.2, noise_portion = 1.0, **kwargs) :
        super().__init__(*args, transform = transform, **kwargs)

        # Generate random noise
        self.noise  = torch.randn(self.data.shape).transpose(1, 3)
        self.noise *= noise_std

        # Keep portion of clean samples
        n         = self.noise.size(0)
        n_clean   = n - int(n * noise_portion)
        idx_clean = torch.randperm(n)[:n_clean]
        self.noise[idx_clean] *= 0
    
    def __getitem__(self, idx) :
        img, target = self._get_clean(idx)
        img = img + self.noise[idx]
        return img, target
    
    def _get_clean(self, idx) :
        return super().__getitem__(idx)
    
class CINIC10(ImageFolder) :
    url = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz'

    def __init__(self, root, split = 'train', download = False, samples_per_class = 5000, **kwargs) :
        # Download dataset
        self.root = Path(root)
        self.split = split
        self.path = self.root / self.split
        if download :
            self.download()
        if not self.path.exists() :
            raise RuntimeError(f'dataset not found in location {root}')
        super().__init__(self.path, **kwargs)

        # Load all images into memory to prevent excessive file system usage
        self.data = []
        self.targets = []
        class_distr = np.zeros(10)
        for path, target in self.samples :
            # Limit number of samples per class
            if class_distr[target] < samples_per_class :
                sample = self.loader(path)
                self.data.append(sample)
                self.targets.append(target)
                class_distr[target] += 1
        self.data = np.stack(self.data)
    
    def __getitem__(self, index) :
        img = self.data[index]
        target = self.targets[index]

        img = Image.fromarray(img)
        if self.transform is not None :
            img = self.transform(img)
        return img, target
    
    def __len__(self) :
        return self.data.shape[0]
    
    def download(self) :
        if not self.path.exists() :
            download_and_extract_archive(self.url, self.root)
    
class YIFAR10(ImageFolder) :

    def __init__(self, root, split = 'train', samples_per_class = 5000, **kwargs) :
        # Download dataset
        self.root = Path(root)
        self.split = split
        self.path = self.root / self.split
        if not self.path.exists() :
            raise RuntimeError(f'dataset not found in location {root}')
        super().__init__(self.path, **kwargs)

        # Load all images into memory to prevent excessive file system usage
        self.data = []
        self.targets = []
        class_distr = np.zeros(10)
        for path, target in self.samples :
            # Limit number of samples per class
            if class_distr[target] < samples_per_class :
                sample = self.loader(path)
                self.data.append(sample)
                self.targets.append(target)
                class_distr[target] += 1
        self.data = np.stack(self.data)
    
    def __getitem__(self, index) :
        img = self.data[index]
        target = self.targets[index]

        img = Image.fromarray(img)
        if self.transform is not None :
            img = self.transform(img)
        return img, target
    
    def __len__(self) :
        return self.data.shape[0]