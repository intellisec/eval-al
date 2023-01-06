# Randomness is the Root of All Evil: Reliable Evaluation of Deep Active Learning


Using deep neural networks for active learning (AL) poses significant challenges for the stability and the reproducibility of experimental results. Inconsistent settings continue to be the root causes for contradictory conclusions and in worst cases, for incorrect appraisal of methods. Our community is in search of a unified framework for exhaustive and fair evaluation of deep active learning. We provide just such a framework, one which is built upon systematically fixing, containing and interpreting sources of randomness. We isolate different influence factors, such as neural-network initialization or hardware specifics, to assess their impact on the learning performance. We then use our framework to analyze the effects of basic AL settings, such as the query-batch size and the use of subset selection, and different datasets on AL performance. Our findings enable us to derive specific recommendations for the reliable evaluation of deep active learning, thus helping advance the community toward a more normative evaluation of results.

For further details please consult the [conference publication](https://intellisec.de/pubs/2023-wacv.pdf).

Below you see an overview of the influence factors considered by our work in comparison to related work. Up to now, the community has only observed isolated effects that become apparent in the evaluation of AL. We are the first to systematically tie these effects to various sources of randomness.

<img src="https://intellisec.de/research/eval-al/overview.svg"  width="900" /><br />


As an example, the used GPU model can change the ranking of AL methods despite enforcing deterministic computations in the learning framework. The lower the averaged results in the "pair-wise penalty matrix" the better.

<img src="https://intellisec.de/research/eval-al/results-gpu.svg" width="700" /><br />


## Publication

A detailed description of our work has been presented at the ([WACV 2023](https://wacv2023.thecvf.com/)) in January 2023. If you would like to cite our work, please use the reference as provided below:

```
@InProceedings{Ji2023Randomness,
author    = {Yilin Ji and Daniel Kaestner and Oliver Wirth
and Christian Wressnegger},
booktitle = {Proc. of the {IEEE} Winter Conference on Applications
of Computer Vision ({WACV})},
title     = {Randomness is the Root of All Evil:
More Reliable Evaluation of Deep Active Learning},
year      = {2023},
month     = jan
}
```

A preprint of the paper is available [here](https://intellisec.de/pubs/2023-wacv.pdf).

## Code

All experiments from the paper can be easily run based on the  bash files in `./experiments_paper`. The log files and results (including the accuracy results and last state) will be save in `./experiments_paper/log/ and ./experiments_paper/results_txt/`

Before you starts, make sure to correctly set up a Python environment. We have been using Python 3.9 with PyTorch 1.8 on CUDA 11. The recommened environment can be easily built via [Conda](https://conda.io):

```
conda env create -f requirements.yml
conda activate torch
```


#### Dowload Dataset

If you want to run a single experiment, you can skip this step, the dataset will be automatically download. For smoothly running a batch of experiments, please firstly download dataset by running:  
```
python3 load_dataset.py
```

#### Run Experiments

All configuration files are in `./conf`. For running a specific experiment simply call the `al_trainer.py` script with the experiment's configuration file. For instance:

```
python3 al_trainer.py -c conf/subset/entropy/entropy_Resnet18_cifar10_I1000_B1000_c26.yml
```

Additionally, you can run specific sets of experiments, for which we have prepared individual bash scripts. In the following, we provide a list of available scripts:

* **Influence of Query Batch Size**  
  `bash ./experiments_paper/batch_size.sh`  
  The query batch size (1000, 2000, 4000) can be set in the bash script directly.
  
* **Influence of Model Initialization and Init-Set Selection**  
  `bash ./experiments_paper/randomness.sh`

* **Influence of Subset Sampling**  
  `bash ./experiments_paper/subset.sh`

* **Influence of Warm Start**  
  `bash ./experiments_paper/warm_cold.sh`

* **Performances on imbalanced CIFAR10**  
  `bash ./experiments_paper/imbalanced.sh`  
  You can set the imbalanced ratio (10 or 100) in the bash script directly

* **Performance on TinyImageNet**  
  `bash ./experiments_paper/imbalanced.sh`


#### Evaluate Results

For evaluating the results of the experiments run using the script above, you can use the `plot_heatmaps.py` script. It produces a pair-wise penalty matrix (PPM) visualized as heatmaps as shown in our [paper](https://intellisec.de/pubs/2023-wacv.pdf). Using the program arguments you can choose the set of influence factor to include:
```
cd ./experiments_paper/plot
python3 plot_heatmaps.py -t Resnet18 -d cifar10 -i 1000 -b 2000 -c 13
```

The  above example evaluates Resnet18 learned on the CIFAR-10 dataset, with 1,000 samples as initial labeled pool (init set) and 2,000 sample for each query batch over 13 cycles.

The plots will be saved in `./experiments_paper/plot/plot_results/`.

#### Add Additional AL Methods

The current version of our framework includes
* Badge ([Ash et al., ICLR 2020](https://arxiv.org/abs/1906.03671))
* BALD  ([Houlsby et al., CoRR 2011](https://arxiv.org/abs/1112.5745))
* Entropy ([Settles and Craven, EMNLP 2008](https://aclanthology.org/D08-1112.pdf))
* CoreSet ([Sener and Savarese, ICLR 2018](https://arxiv.org/abs/1708.00489v4))
* LLOSS ([Yoo and Kweon, CVPR 2019](https://arxiv.org/abs/1905.03677))
* ISAL ([Brodersen et al., ICPR 2010](https://ieeexplore.ieee.org/document/5597285))
* LC ([Lewis and Gale, SIGIR 1994](https://arxiv.org/abs/cmp-lg/9407020))

implemented as python scripts in the `queries_strategies` folder. More AL startegies can be added in that folder. Do not forget to also list the AL method in `al_trainer.py`
