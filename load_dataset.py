import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10, FashionMNIST, SVHN, MNIST
from data.tinyimagenet import TinyImageNet
from data.openml import OpenMLDataset
from data.unclean import NoisyCIFAR10, CINIC10, YIFAR10
from data.imdb import ImdbDataset
from random import randint
import random

class MyDataset(Dataset):
    def __init__(self, dataset_name, train_flag, transf, cfgs = None):
        self.dataset_name = dataset_name
        if self.dataset_name == "cifar10":
            self.cifar10 = CIFAR10('../cifar10', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "mnist":
            self.mnist = MNIST('../mnist', train=train_flag,
                                    download=True, transform=transf)
        if self.dataset_name == "cifar100":
            self.cifar100 = CIFAR100('../cifar100', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "fashionmnist":
            self.fmnist = FashionMNIST('../fashionMNIST', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "svhn":
            self.svhn = SVHN('../svhn', split="train", 
                                    download=True, transform=transf)
        if self.dataset_name == "tinyimagenet":
            self.tinyimagenet = TinyImageNet('../tinyimagenet', split="train", 
                                    download=True, transform=transf)
        if self.dataset_name == "openml":
            self.openml = OpenMLDataset(train = train_flag)
        if self.dataset_name == "cifar10noise":
            self.cifar10noise = NoisyCIFAR10('../cifar10', train=train_flag,  download=True, transform=transf,
                                      noise_std = cfgs.DATASET_NOISE_STD, noise_portion = cfgs.DATASET_NOISE_PORTION)
        if self.dataset_name == "cinic10":
            self.cinic10 = CINIC10('../cinic10', split='train', download=True, transform=transf)
        if self.dataset_name == "imdb":
            self.imdb = ImdbDataset('../imdb', split = 'train', download = True)
        if self.dataset_name == "yifar10":
            self.yifar10 = YIFAR10('../yifar10', split='train', transform=transf)


    def __getitem__(self, index):
        if self.dataset_name == "cifar10":
            data, target = self.cifar10[index]
        if self.dataset_name == "mnist":
            data, target = self.mnist[index]
        if self.dataset_name == "cifar100":
            data, target = self.cifar100[index]
        if self.dataset_name == "fashionmnist":
            data, target = self.fmnist[index]
        if self.dataset_name == "svhn":
            data, target = self.svhn[index]
        if self.dataset_name == "tinyimagenet":
            data, target = self.tinyimagenet[index]
        if self.dataset_name == "openml":
            data, target = self.openml[index]
        if self.dataset_name == "cifar10noise":
            data, target = self.cifar10noise[index]
        if self.dataset_name == "cinic10":
            data, target = self.cinic10[index]
        if self.dataset_name == "imdb":
            data, target = self.imdb[index]
        if self.dataset_name == "yifar10":
            data, target = self.yifar10[index]
        return data, target, index

    def __len__(self):
        if self.dataset_name == "cifar10":
            return len(self.cifar10)
        if self.dataset_name == "mnist":
            return len(self.mnist)
        elif self.dataset_name == "cifar100":
            return len(self.cifar100)
        elif self.dataset_name == "fashionmnist":
            return len(self.fmnist)
        elif self.dataset_name == "svhn":
            return len(self.svhn)
        elif self.dataset_name == "tinyimagenet":
            return len(self.tinyimagenet)
        elif self.dataset_name == "openml":
            return len(self.openml)
        elif self.dataset_name == "cifar10noise":
            return len(self.cifar10noise)
        elif self.dataset_name == "cinic10":
            return len(self.cinic10)
        elif self.dataset_name == "imdb":
            return len(self.imdb)
        elif self.dataset_name == "yifar10":
            return len(self.yifar10)
##
  

# Data
def load_dataset(dataset, cfgs = None):
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])


    if dataset == 'cifar10': 
        data_train = CIFAR10('../cifar10', train=True, download=True, transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)
        NO_CLASSES = 10
    elif dataset == 'mnist':
        data_train = MNIST('../mnist', train=True, download=True, transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test = MNIST('../mnist', train=False, download=True, transform=T.Compose([T.ToTensor()]))
        NO_CLASSES = 10
        # adden = ADDENDUM
    elif dataset == 'cifar10im': 
        data_train = CIFAR10('../cifar10', train=True, download=True, transform=train_transform)
        #data_unlabeled   = CIFAR10('../cifar10', train=True, download=True, transform=test_transform)
        targets = np.array(data_train.targets)
        #NUM_TRAIN = targets.shape[0]
        classes, class_counts = np.unique(targets, return_counts=True)
        nb_classes = len(classes)

        # Reduce imbalanced classes' samples
        imb_classes = np.arange(5) # as defined in TA-VAAL paper
        b_clasees = np.arange(5,10)
        sum_imbclasses = class_counts[b_clasees].sum() // cfgs.DATASET_IMB_RATIO
        num_imbclasses = [0]*5
        random.seed(1993)
        for i in range(sum_imbclasses):
            num_imbclasses[randint(0,sum_imbclasses) % 5] += 1
 
        class_counts[imb_classes] = np.array(num_imbclasses) 
        # print("class_counts after {}".format(class_counts))
        # class_counts[imb_classes] //= cfgs.DATASET_IMB_RATIO

        # print(class_counts)
        class_idxs = [np.where(targets == i)[0] for i in range(nb_classes)]
        imb_class_idx = [class_id[:class_count] for class_id, class_count in zip(class_idxs, class_counts)]
        imb_class_idx = np.hstack(imb_class_idx)
        no_train = imb_class_idx.shape[0]
        data_train.targets = targets[imb_class_idx]
        data_train.data = data_train.data[imb_class_idx]
        data_unlabeled = MyDataset('cifar10', True, test_transform)
        data_unlabeled.cifar10.targets = targets[imb_class_idx]
        data_unlabeled.cifar10.data = data_unlabeled.cifar10.data[imb_class_idx]
        data_test  = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)
        NO_CLASSES = 10
    elif dataset == 'cifar100':
        data_train = CIFAR100('../cifar100', train=True, download=True, transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR100('../cifar100', train=False, download=True, transform=test_transform)
        NO_CLASSES = 100
    elif dataset == 'fashionmnist':
        data_train = FashionMNIST('../fashionMNIST', train=True, download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))
        data_test  = FashionMNIST('../fashionMNIST', train=False, download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        NO_CLASSES = 10
    elif dataset == 'svhn':
        data_train = SVHN('../svhn', split='train', download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))
        data_test  = SVHN('../svhn', split='test', download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        NO_CLASSES = 10
    elif dataset == 'tinyimagenet':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        data_train = TinyImageNet('../tinyimagenet', split='train', download=True, transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = TinyImageNet('../tinyimagenet', split='val', download=True, transform=test_transform)
        NO_CLASSES = 200
    elif dataset == 'openml' :
        data_train = OpenMLDataset(train = True)
        data_unlabeled = MyDataset(dataset, True, None)
        data_test  = OpenMLDataset(train = False)
        NO_CLASSES = data_train.n_classes
    elif dataset == 'cifar10noise': 
        data_train     = NoisyCIFAR10('../cifar10', train=True,  download=True, transform=train_transform,
                                      noise_std = cfgs.DATASET_NOISE_STD, noise_portion = cfgs.DATASET_NOISE_PORTION)
        data_unlabeled = MyDataset(dataset, True, test_transform, cfgs)
        data_test      =      CIFAR10('../cifar10', train=False, download=True, transform=test_transform)

        # Use same noise for train and unlabeled set
        data_unlabeled.noise = data_train.noise
        NO_CLASSES = 10
    elif dataset == 'cinic10':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835])
        ])

        test_transform2 = T.Compose([
            T.ToTensor(),
            T.Normalize([0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835])
        ])

        data_train     = CINIC10('../cinic10', split='train', download=True, transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform2)
        data_test      = CIFAR10('../cifar10', train=False,   download=True, transform=test_transform)
        NO_CLASSES = 10
    elif dataset == 'imdb' :
        data_train     = ImdbDataset('../imdb', split = 'train', download = True)
        data_unlabeled = MyDataset(dataset, True, None)
        data_test      = ImdbDataset('../imdb', split = 'test',  download = True)
        NO_CLASSES = data_train.n_classes
    elif dataset == 'yifar10':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.5039, 0.4880, 0.4425], [0.2055, 0.2011, 0.2045])
        ])

        test_transform2 = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5039, 0.4880, 0.4425], [0.2055, 0.2011, 0.2045])
        ])

        data_train     = YIFAR10('../yifar10', split='train', transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform2)
        data_test      = CIFAR10('../cifar10', train=False,   download=True, transform=test_transform)
        NO_CLASSES = 10
    return data_train, data_unlabeled, data_test, NO_CLASSES

if __name__ == "__main__":
    cifar10_tr,_,_,_ = load_dataset('cifar10')
    cifar100_tr,_,_,_ = load_dataset('cifar100')
    tinyimagenet,_,_,_ = load_dataset('tinyimagenet')
