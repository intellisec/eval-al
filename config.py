from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
import os
import os.path as osp
import numpy as np

"""config system.
This file specifies default config options. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options.
"""


class AttrDict(dict):
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value


__C = AttrDict()

cfg = __C

# Check the training setting for each method for original paper
__C.METHOD = 'CoreSet'
__C.DATASET = 'cifar100'
__C.TASK_LEARNER = 'resnet18'
__C.FULLSET = False #if True, use SUBSET in each query interation
__C.TOTAL_DATASET = False #if True, labeled all data points initially and train them at once
__C.CYCLES = 18 
__C.TRIALS = 5
__C.INIT_BUDGET = 1000
__C.BUDGET = 1000 
__C.SUB_PRE = 10000 #num of subset
__C.BATCH = 128 # general batch size for data loader
__C.NUM_TRAIN = 50000 # size of the original data pool
__C.RANDOM_SET_SEED = None
__C.RANDOM_SET_SEED_ORDER = None
__C.RANDOM_MODEL_PARAM_SEED = None
__C.BUDGET = 1000
__C.STRAT_SAMPLING = False # use stratified sampling for initially labeled pool, useful for INIT_BUDGET=10
__C.DATASET_NOISE_STD = 0.2 # standard deviation of gaussian noise in cifar10noise dataset
__C.DATASET_NOISE_PORTION = 0.8 # portion of noisy samples in cifar10noise dataset
__C.DATASET_IMB_RATIO = 1 # ratio between number of samples in normal and reduced classes in cifar10im dataset
__C.results_folder = 'results_paper/batch_size_1/'
__C.warm_start = True


__C.GCN = AttrDict()
__C.GCN.DROPOUT_RATE = 0.3
__C.GCN.S_MARGIN = 0.1
__C.GCN.HIDDEN_UNITS = 128
__C.GCN.LAMBDA_LOSS = 1.2
__C.GCN.EPOCH_GCN = 200
__C.GCN.LR_GCN = 1e-3
__C.GCN.WDECAY_GCN = 5e-4

__C.VAAL = AttrDict() 
__C.VAAL.EPOCHV = 100 # training epochs for vae and diminator in VAAL, TA-VAAL, Inf-VAAL
__C.VAAL.Z_DIM = 32
__C.VAAL.LR = 5e-4

__C.BALD = AttrDict()
__C.BALD.DROPOUT_ITER = 25


__C.OPTIMIZER_BACKBONE = AttrDict()
__C.OPTIMIZER_BACKBONE.NAME = 'SGD'
__C.OPTIMIZER_BACKBONE.MOMENTUM =  0.9
__C.OPTIMIZER_BACKBONE.WDECAY = 5e-4
__C.OPTIMIZER_BACKBONE.LR_SGD = 1e-1
__C.OPTIMIZER_BACKBONE.LR_ADAM = 1e-3
__C.OPTIMIZER_BACKBONE.MILESTONES = [160, 240]
__C.OPTIMIZER_BACKBONE.EPOCH_TL = 200 # train epoch for task learner

__C.OPTIMIZER_MODULE = AttrDict()
__C.OPTIMIZER_MODULE.NAME = 'SGD'
__C.OPTIMIZER_MODULE.MOMENTUM =  0.9
__C.OPTIMIZER_MODULE.WDECAY = 5e-4
__C.OPTIMIZER_MODULE.LR = 1e-1 #the original changes are 1e-2
__C.OPTIMIZER_MODULE.MILESTONES = [160, 240]
__C.OPTIMIZER_MODULE.MARGIN = 1.0 #xi
__C.OPTIMIZER_MODULE.WEIGHT = 1.0 #lambda
__C.OPTIMIZER_MODULE.EPOCHL =120 #for lloss and TA-VAAL, after 120 epoch should add the predicted loss for training 

def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), "Argument `a` must be an AttrDict"
    assert isinstance(b, AttrDict), "Argument `b` must be an AttrDict"

    for k, v_ in a.items():
        full_key = ".".join(stack) + "." + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError("Non-existent config key: {}".format(full_key))

        v = _decode_cfg_value(v_)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v



def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml

    with open(filename, "r") as f:
        yaml_cfg = AttrDict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)
 
    return cfg


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
  
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_a, int) and (value_b is None):
        value_a = value_a
    else:
        raise ValueError(
            "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
            "key: {}".format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
