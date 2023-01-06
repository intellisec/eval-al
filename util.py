import sys 
import torch

from models.target_models import vgg16, vgg16_bn, ResNet18, ResNet18fm, MLP
from models.target_models_offcially import resnext50_32x4d
# from queries_strategies import RandomSelection, Entropy, BALD, Badge,CoreSet
from queries_strategies import *


def Build_model(data_train, method, NO_CLASSES, device, cfgs, args, model_param_seed=None, recover=False):
    # Model - create new instance for every cycle so that it resets
    with torch.cuda.device(device):
        if cfgs.TASK_LEARNER=="resnet18":
            if cfgs.DATASET == "fashionmnist" or cfgs.DATASET == 'mnist':
                model   = ResNet18fm(num_classes=NO_CLASSES,
                                     fixed_model_parameter_seed=model_param_seed).cuda()
            else:
                print(NO_CLASSES)
                model    = ResNet18(num_classes=NO_CLASSES,
                                    fixed_model_parameter_seed=model_param_seed).cuda()
                print('this time we use pytorch resnet18')
        elif cfgs.TASK_LEARNER=="resnext50":
            print('use ResNext50 as task learner')
            model = resnext50_32x4d(num_classes=NO_CLASSES).cuda()
            torch.manual_seed(model_param_seed)


        elif cfgs.TASK_LEARNER=="vgg16":
            print('use vgg16 as task learner')
            model = vgg16(num_classes=NO_CLASSES).cuda()

        elif cfgs.TASK_LEARNER=="vgg16_bn":
            model = vgg16_bn(num_classes=NO_CLASSES).cuda()

        elif cfgs.TASK_LEARNER=='mlp':
            model = MLP(num_classes = NO_CLASSES).cuda()
            if cfgs.DATASET in ('openml', 'imdb') :
                model = MLP(num_classes = NO_CLASSES, input_size = (data_train.n_features, )).cuda()
        else:
            print('choose a valid acquisition function', flush=True)
            raise ValueError

        if method in ('lloss', 'TA-VAAL'):
            #loss_module = LossNet(feature_sizes=[16,8,4,2], num_channels=[128,128,256,512]).cuda()
            if cfgs.TASK_LEARNER=='vgg16':
                loss_module = LossNet(feature_sizes=[32,16,8,8], num_channels=[64, 128, 256, 256]).cuda()
            elif cfgs.TASK_LEARNER=='mlp':
                loss_module = LossNet(feature_sizes=[1], num_channels=[model.get_embedding_dim()]).cuda()
            elif cfgs.DATASET=='tinyimagenet' :
                loss_module = LossNet(feature_sizes=[64, 32, 16, 8]).cuda()
            else:
                loss_module = LossNet().cuda()
    print('build model')
    if recover:
        backbone_state = args.state.split('state')[0]+'target_model.pt'
        print('recover target model from the pickled file : {} '.format(backbone_state))
        model.load_state_dict(torch.load(backbone_state))
        if method in ('lloss', 'TA-VAAL'):
            lossmodule_state = args.state.split('state')[0]+'loss_model.pt'
            print('recover loss module from the pickled file : {}'.format(lossmodule_state))
            loss_module.load_state_dict(torch.load(lossmodule_state))

    models      = {'backbone': model}
    if method in ('lloss', 'TA-VAAL'):
        models = {'backbone': model, 'module': loss_module}
    return models

def query_strategy(method, model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device):
    if method == 'Random':
        strategy = RandomSelection(model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device)
    elif method == 'Entropy' or method == 'entropy':
        strategy = Entropy(model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device)
    elif method == 'BALD':
        strategy = BALD(model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device)
    elif method == 'CoreSet':
        strategy = CoreSet(model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device)
    elif method == 'Badge':
        strategy = Badge(model, data_unlabeled, NO_CLASSES, test_loader, cfgs, device)
    else:
        print('choose a valid querying function', flush=True)
        raise ValueError
    return strategy

def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress
    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()
