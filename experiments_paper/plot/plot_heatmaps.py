from pathlib import Path

import os
from argparse import ArgumentParser
from preprocessing import batch_size, deterministic, hardware, init_set, model_initialization,tinyimagenet

# Argument parser for command line use
parser = ArgumentParser(description = 'Plot and compare AL strategies for a given training setting')
parser.add_argument('-t', '--task_learner', type = str, required = True,  help = 'Name of task learning model')
parser.add_argument('-d', '--dataset',      type = str, required = True,  help = 'Learning dataset')
parser.add_argument('-i', '--init_budget',  type = int, required = True,  help = 'Initial labeled pool size')
parser.add_argument('-b', '--budget',       type = int, required = True,  help = 'Query budget per cycle')
parser.add_argument('-c', '--cycles',       type = int, required = False,  help = 'Number of query iterations')

def plot_heatmap_batch_size(task_learner, dataset, init_budget, results_base_path, plots_base_path):  
    results_path = results_base_path / f'batch_size'
    plots_path = plots_base_path / f'batch_size'
    os.makedirs(plots_path, exist_ok=True)
    batch_size.plot_setting(task_learner, dataset, init_budget, results_path, plots_path)

def plot_heatmap_model_initialization(task_learner, dataset, init_budget, budget, cycles,results_base_path, plots_base_path):
    results_path = results_base_path / f'init_set'
    plots_path = plots_base_path / f'model_initialization'
    os.makedirs(plots_path, exist_ok=True)
    model_initialization.plot_setting(task_learner, dataset, init_budget, budget, cycles,results_path, plots_path)

def plot_heatmap_init_set(task_learner, dataset, init_budget, budget, cycles, results_base_path, plots_base_path):
    results_path = results_base_path / f'init_set'
    plots_path = plots_base_path / f'init_set'
    os.makedirs(plots_path, exist_ok=True)
    init_set.plot_setting(task_learner, dataset, init_budget, budget, cycles,results_path, plots_path)

def plot_heatmap_deterministic(task_learner, dataset, init_budget, budget, cycles, results_base_path, plots_base_path):
    results_path = results_base_path / f'NonDeterministic'
    plots_path = plots_base_path / f'NonDeterministic'
    os.makedirs(plots_path, exist_ok=True)
    deterministic.plot_setting(task_learner, dataset, init_budget, budget, cycles,results_path, plots_path)

def plot_heatmap_hardware(task_learner, dataset, init_budget, budget, cycles,results_base_path, plots_base_path):
    results_path = results_base_path / f'hardware'
    plots_path = plots_base_path / f'hardware'
    os.makedirs(plots_path, exist_ok=True)
    hardware.plot_setting(task_learner, dataset, init_budget, budget, cycles,results_path, plots_path)

# def plot_heatmap_subset(task_learner, dataset, init_budget, results_base_path, plots_base_path):
#     ...

# def plot_heatmap_cifar10im(task_learner, dataset, init_budget, results_base_path, plots_base_path):
    ...

def plot_heatmap_tinyimagenet(task_learner, dataset, init_budget, results_base_path, plots_base_path):
    results_path = results_base_path / f'tinyimagenet'
    plots_path = plots_base_path / f'tinyimagenet'
    os.makedirs(plots_path, exist_ok=True)
    thiyimagenet.plot_setting(task_learner, dataset, init_budget, budget, cycles,results_path, plots_path)

def plot_heatmaps(task_learner, dataset, init_budget, budget, cycles):
    plots_base_path = Path('./plot_results/')
    results_base_path = Path('./raw_results/')
    os.makedirs(plots_base_path, exist_ok=True)
    plot_heatmap_hardware(task_learner, dataset, init_budget, budget, cycles, results_base_path, plots_base_path)
    plot_heatmap_batch_size(task_learner, dataset, init_budget, results_base_path, plots_base_path)
    plot_heatmap_model_initialization(task_learner, dataset, init_budget, budget, cycles,results_base_path, plots_base_path)
    plot_heatmap_init_set(task_learner, dataset, init_budget, budget, cycles, results_base_path, plots_base_path)
    plot_heatmap_deterministic(task_learner, dataset, init_budget, budget, cycles, results_base_path, plots_base_path)


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    plot_heatmaps(**args)