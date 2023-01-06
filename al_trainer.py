'''
Reliable Evaluation of Deep Active Learning
'''

# Python
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import torchvision.models as torchmodels
import argparse
import datetime, shutil, psutil
import pickle
from pathlib import Path
# Custom
from models.target_models import vgg16, vgg16_bn, ResNet18, ResNet18fm, MLP
from models.target_models_offcially import resnext50_32x4d;
from config import cfg_from_file
from train_test import train, test
from random_seed_generator import RandomSeedGenerator
from models.query_models import LossNet
from data.sampler import stratified_sampling
from load_dataset import load_dataset
from util import Build_model, query_strategy


parser = argparse.ArgumentParser()
parser.add_argument("-c","--config_file",type=str, default='./configs/TA_VAAL_R18_I1000_B1000_c40.yml', 
                    help="path of the config file")
parser.add_argument("-s", "--state", type=str, help="Recover state from pickled file")
parser.add_argument("-cuda","--cuda_visible_device", type=int, default=0, help="CUDA_VISIBLE_DEVICES")
# parser.add_argument("-pre_train", "--pretrained_model", type=str, help="initialization with pretrained model")
# parser.add_argument("-warm", "--warm_start", type=bool, default=False, help="enable warm start during iterations")
parser.add_argument("-vis", "--tensorboard_vis", type=bool, default=False, help="enable visulization in tensorboard")
parser.add_argument("-dm", "--log_data_map", type=bool, default=False, help="log accuracies of each epoch for a datamap")
parser.add_argument("-det", "--deterministic", type=bool, default=False, help="train deterministic")

args = parser.parse_args()

# Main
if __name__ == '__main__':
    '''
    method_type: 'Random', 'CoreSet', 'Badge', 'Entropy', 'BALD'
    datasets: 'cifar10, cifar100, cifa10im, tinyimagenet'
    '''

    methods = ['Random', 'CoreSet', 'Badge', 'Entropy', 'BALD', 'entropy']
    datasets = ['cifar10', 'cifar100', 'tinyimagenet', 'cifar10im','mnist']

    # Parse config file
    config_file = args.config_file

    cfgs = cfg_from_file(args.config_file)
    method = cfgs.METHOD
    device = args.cuda_visible_device
    assert method in methods, 'No method %s! Try options %s'%(method, methods)
    assert cfgs.DATASET in datasets, 'No dataset %s! Try options %s'%(cfgs.DATASET, datasets)

    # base_dir = os.path.dirname(os.path.realpath(__file__)) 
    # results_folder = os.path.join(base_dir, cfgs.results_folder)   
    # print("results_folder: {}".format(results_folder))
    results_path = os.path.join(cfgs.results_folder, cfgs.METHOD.replace('-', '_'))
    if not os.path.exists(results_path) :
        os.makedirs(results_path)
    print("results_path: {}".format(results_path))

    print("Dataset: %s"%cfgs.DATASET)
    print("Method type:%s"%cfgs.METHOD)
    print('Task learner:%s'%cfgs.TASK_LEARNER)
    print('Full set:', bool(cfgs.FULLSET))

    # Starting trial and cycle number
    trial0 = 0
    cycle0 = 0

    # Initialize the random seed generator for the initial set and generate seeds for each trial
    rand_set_seed_gen = RandomSeedGenerator(cfgs.RANDOM_SET_SEED)
    random_set_seeds = rand_set_seed_gen.get_k_random_numbers(cfgs.TRIALS)

    # Initialize the random seed generator for the model parameters and generate seeds for each trial
    rand_model_param_seed_gen = RandomSeedGenerator(cfgs.RANDOM_MODEL_PARAM_SEED)
    random_model_param_seeds = rand_model_param_seed_gen.get_k_random_numbers(cfgs.TRIALS)

    # Recover state from file
    recover = False
    recover_state = {}
    if args.state :
        recover = True
        with open(args.state, 'rb') as f :
            recover_state = pickle.load(f)
        # Overwrite starting trial and cycle
        trial0 = recover_state['trial']
        cycle0 = recover_state['cycle'] + 1
        random_set_seeds = recover_state['random_seed_states']
        random_model_param_seeds = recover_state['random_model_states']


    # Setup result file
    ex_setting = config_file.split('/')[-1].split('.')[0]
    write_mode = 'a' if recover else 'w'
    results = open(os.path.join(results_path, ex_setting+'.txt'), write_mode)

    # Check if full train 
    if cfgs.TOTAL_DATASET:
        print('Train TL once with all data points', flush=True)
        cfgs.TRIALS = 1
        cfgs.CYCLES = 1

    # Load training and testing dataset
    data_train, data_unlabeled, data_test, NO_CLASSES = load_dataset(cfgs.DATASET, cfgs)
    for trial in range(trial0, cfgs.TRIALS):
        # set seed for random init set selection
        if cfgs.RANDOM_SET_SEED:
            if cfgs.RANDOM_SET_SEED_ORDER:
                init_set_seed = random_set_seeds[cfgs.RANDOM_SET_SEED_ORDER]
            else:
                init_set_seed = random_set_seeds[trial]
            random.seed(init_set_seed)
            print("Initial Random Set Seed:" + str(init_set_seed))   

        # construct labeled dataloader and test data loader
        indices = list(range(cfgs.NUM_TRAIN))
        random.shuffle(indices)

        if cfgs.TOTAL_DATASET:
            labeled_set   = indices

        elif recover:
            # Reset recover flag to ensure only loading state once
            labeled_set   = recover_state['labeled_set']
            unlabeled_set = recover_state['unlabeled_set']

        elif cfgs.STRAT_SAMPLING :
            labeled_set   = stratified_sampling(indices, data_train, cfgs.INIT_BUDGET, NO_CLASSES)
            unlabeled_set = [x for x in indices if x not in labeled_set]
       
        else:
            labeled_set   = indices[:cfgs.INIT_BUDGET]
            unlabeled_set = indices[cfgs.INIT_BUDGET:]

        train_loader = DataLoader(data_train, batch_size=cfgs.BATCH,
                                    sampler=SubsetRandomSampler(labeled_set),
                                    pin_memory=True, drop_last=False)
        # val_loader = DataLoader(data_train, batch_size=BATCH,
        #                             sampler=SubsetRandomSampler(val_indices),
        #                             pin_memory=True, drop_last=False)
        test_loader  = DataLoader(data_test, batch_size=cfgs.BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}

        if cfgs.warm_start:
            print('enable warm starting')

            # Load random seeds for the initial model parameters (returns None if no seed is configured)
            model_param_seed = random_model_param_seeds[trial] if cfgs.RANDOM_MODEL_PARAM_SEED else None
            print("Initial Random Model Seed:" + str(model_param_seed))
            models = Build_model(data_train, method, NO_CLASSES, device, cfgs, args, model_param_seed, recover)
            recover = False
            # acc = test(models, method, dataloaders, device, mode='test')
            # print('Trial {}/{} || Cycle {}/{} || model initialization || test acc {}'.format(trial+1, cfgs.TRIALS, 0+1, cfgs.CYCLES, acc))
            # control torch deterministic training 
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

             # initialize query strategy
            strategy = query_strategy(method, models, data_unlabeled, NO_CLASSES, test_loader, cfgs, device)
            strategy.seed_random_init_set(init_set_seed)

        for cycle in range(cycle0, cfgs.CYCLES):

            # Construct subset for selection
            if not cfgs.TOTAL_DATASET:
                random.shuffle(unlabeled_set)
                if not cfgs.FULLSET:
                    # Apply subset scheme
                    subset = unlabeled_set[:cfgs.SUB_PRE] 
                else:
                    subset = unlabeled_set
                print('size of ulb pool {}'.format(len(subset)))

            # Model - create new instance for every cycle so that it resets, when it's cold start
            if not cfgs.warm_start:
                print('train with cold start between AL rounds')

                recover = False
                model_param_seed = random_model_param_seeds[trial] if cfgs.RANDOM_MODEL_PARAM_SEED else None
                print("Initial Random Model Seed:" + str(model_param_seed))
                models = Build_model(data_train, method, NO_CLASSES, device, cfgs, args, model_param_seed, recover)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                acc = test(models, method, dataloaders, device, mode='test')
                print('Trial {}/{} || Cycle {}/{} || model initialization || test acc {}'.format(trial+1, cfgs.TRIALS, cycle+1, cfgs.CYCLES, acc))
                # initialize query strategy
                strategy = query_strategy(method, models, data_unlabeled, NO_CLASSES, test_loader, cfgs, device)
                strategy.seed_random_init_set(init_set_seed)

            #Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')
            if cfgs.OPTIMIZER_BACKBONE.NAME=='SGD':
                optim_backbone = optim.SGD(models['backbone'].parameters(), lr=cfgs.OPTIMIZER_BACKBONE.LR_SGD, momentum=cfgs.OPTIMIZER_BACKBONE.MOMENTUM, weight_decay=cfgs.OPTIMIZER_BACKBONE.WDECAY)
                sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=cfgs.OPTIMIZER_BACKBONE.MILESTONES)
                schedulers = {'backbone': sched_backbone}
            else:
                optim_backbone = optim.Adam(models['backbone'].parameters(), lr=cfgs.OPTIMIZER_BACKBONE.LR_ADAM)
                schedulers = {}
            optimizers = {'backbone': optim_backbone}
            
            if method in ('lloss', 'TA-VAAL'):
                optim_module   = optim.SGD(models['module'].parameters(), lr=cfgs.OPTIMIZER_MODULE.LR, momentum=cfgs.OPTIMIZER_MODULE.MOMENTUM, weight_decay=cfgs.OPTIMIZER_MODULE.WDECAY)
                sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=cfgs.OPTIMIZER_MODULE.MILESTONES)
                # optim_module = optim.Adam(models['module'].parameters(), lr=1e-4)
                optimizers = {'backbone': optim_backbone, 'module': optim_module}
                if cfgs.OPTIMIZER_BACKBONE.NAME =='SGD':
                    schedulers = {'backbone': sched_backbone,'module': sched_module}
                else: schedulers = {'module': sched_module}

            # Training and testing
            print('start training ')
            print('cycle {}, num of labeled data {}'.format(cycle, len(labeled_set)))
            
            train(models, method, criterion, optimizers, schedulers, dataloaders, device, cfgs, cycle, ex_setting, args.log_data_map)
            acc = test(models, method, dataloaders, device, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, cfgs.TRIALS, cycle+1, cfgs.CYCLES, len(labeled_set), acc))
            np.array([method, trial+1, cfgs.TRIALS, cycle+1, cfgs.CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")
            if cfgs.DATASET=='cifar10im':
                # save current model parameters each model to calculate the imbalanced acc
                models_path = os.path.join(results_path, 'models')
                if not os.path.exists(models_path) :
                    os.makedirs(models_path)
                torch.save(models['backbone'].state_dict(), os.path.join(results_path, ex_setting+'_T'+str(trial+1)+'_c'+str(cycle+1)+'_target_model.pt'))

            if cycle == (cfgs.CYCLES-1):
                # Reached final training cycle
                print("Finished.")
                results.write('\n')
                break

            # Measure query time
            start_t = datetime.datetime.now()
            mem_cpu = psutil.virtual_memory().used / 1e9
            mem_gpu = torch.cuda.memory_allocated(device) / 1e9

            # Querying 
            if len(subset)>cfgs.BUDGET:
                strategy.feed_current_state(cycle, subset, labeled_set)
                arg = strategy.query()
            
                # Update the labeled dataset and the unlabeled dataset, respectively
                labeled_set += list(torch.tensor(subset)[arg][-cfgs.BUDGET:].numpy())
                listd = list(torch.tensor(subset)[arg][:-cfgs.BUDGET].numpy()) ### remaining from the subset
                if not cfgs.FULLSET:
                    unlabeled_set = listd + unlabeled_set[cfgs.SUB_PRE:] ### Unselected samples from current subset +
                else:
                    unlabeled_set = listd
            else:
                labeled_set += subset
                unlabeled_set = []

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=cfgs.BATCH,
                                                sampler=SubsetRandomSampler(labeled_set),
                                                pin_memory=True)

            # Save current state
            with open(os.path.join(results_path, ex_setting+'_latest_state.pkl'), 'wb') as f :
                pickle.dump({
                    'trial':         trial,
                    'cycle':         cycle,
                    'labeled_set':   labeled_set,
                    'unlabeled_set': unlabeled_set,
                    'random_seed_states': random_set_seeds,
                    'random_model_states': random_model_param_seeds
                }, f, pickle.HIGHEST_PROTOCOL)

            # save current model parameters
            torch.save(models['backbone'].state_dict(), os.path.join(results_path, ex_setting+'_latest_target_model.pt'))
            if method in ('lloss', 'TA_VAAL'):
                torch.save(models['module'].state_dict(), os.path.join(results_path, ex_setting+'_latest_loss_model.pt'))

            # Stop time
            end_t = datetime.datetime.now()
            took = end_t - start_t
            print(f'Cycle duration: {took}, allocated memory (cpu/gpu): {mem_cpu}GB / {mem_gpu}GB')
            results.write(f" '{took}' '{mem_cpu}' '{mem_gpu}'\n")
            # Ensure output is actually written to file
            results.flush()
            os.fsync(results.fileno())
        # Reset starting cycle to 0
        cycle0 = 0
    results.close()
