from config import *
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
# import models.resnet as resnet
# import gpustat


# Loss Prediction Loss from lloss
def LossPredLoss_ll(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) 
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss

# Loss Prediction Loss from TA_VAAL, ranking loss not the absolute loss in lloss 
def LossPredLoss_ta(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = torch.nn.BCELoss()
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()
    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0)) # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = criterion(diff,one)
    elif reduction == 'none':
        loss = criterion(diff,one)
    else:
        NotImplementedError()
    
    return loss


def test(models, method, dataloaders, device, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()
    
    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            with torch.cuda.device(device):
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            scores, _, _ = models['backbone'](inputs)
            #scores = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    #print('total number for test {}'.format(total))
    return 100 * correct / total

def calc_loss(models, dataloaders, criterion, device, mode):
    models['backbone'].eval()
    losses = []
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            with torch.cuda.device(device):
                inputs = inputs.cuda()
                labels = labels.cuda()
        
            scores, _, _ = models['backbone'](inputs)
            target_loss = criterion(scores, labels)
            losses.append(target_loss)
    l = torch.cat(losses)
    return torch.sum(l)/len(l)

iters = 0
def train_epoch(models, method, criterion, optimizers, schedulers, dataloaders, epoch, epoch_loss, device, MARGIN, WEIGHT, log_data_map, VIS=False):
    models['backbone'].train()
    if method == 'lloss' or method == 'TA-VAAL':
        models['module'].train()
    global iters
    # for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
    total_t = 0
    correct_t = 0
    for (batch_idx, data) in enumerate(dataloaders['train']):
        with torch.cuda.device(device):
            inputs = data[0].cuda()
            labels = data[1].cuda()

        iters += 1

        optimizers['backbone'].zero_grad()
        if method == 'lloss' or method == 'TA-VAAL':
            optimizers['module'].zero_grad()
                   
        scores, _, features = models['backbone'](inputs)
        #scores = models['backbone'](inputs)
        if log_data_map:
            log_datamap_confidence(labels, scores, batch_idx, epoch, cfg)

        if VIS:
            _, preds = torch.max(scores.data, 1)
            total_t += labels.size(0)
            correct_t += (preds == labels).sum().item()
        target_loss = criterion(scores, labels)
        if method in ('lloss','TA-VAAL'):
            if epoch > epoch_loss:
                features = [f.detach() for f in features]
            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            if method == 'lloss':
                m_module_loss   = LossPredLoss_ll(pred_loss, target_loss, margin=MARGIN)
            else:
                m_module_loss   = LossPredLoss_ta(pred_loss, target_loss, margin=MARGIN)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss + WEIGHT * m_module_loss 
        else:
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss

        loss.backward()
        optimizers['backbone'].step()
        if method == 'lloss' or method == 'TA-VAAL':
            optimizers['module'].step()
    if VIS:
      #  print('total num in training {}'.format(total_t))
        return loss, 100 * correct_t / total_t
    else:
        return loss

def train(models, method, criterion, optimizers, schedulers, dataloaders, device, cfgs, cycle, ex_setting, log_data_map):
    print('>> Train a Model.')
    num_epochs = cfgs.OPTIMIZER_BACKBONE.EPOCH_TL
    epoch_loss = cfgs.OPTIMIZER_MODULE.EPOCHL
    MARGIN = cfgs.OPTIMIZER_MODULE.MARGIN
    WEIGHT = cfgs.OPTIMIZER_MODULE.WEIGHT
    best_acc = 0.
    # writer = SummaryWriter(log_dir='./runs/'+ ex_setting + '/exp'+str(cycle))
    for epoch in range(num_epochs): 
        best_loss = torch.tensor([0.5]).cuda()
        train_loss = calc_loss(models, dataloaders, criterion, device, 'train')
        test_loss = calc_loss(models, dataloaders, criterion, device, 'test')
        loss, acc_train = train_epoch(models, method, criterion, optimizers, schedulers, dataloaders, epoch, epoch_loss, device, MARGIN, WEIGHT, log_data_map, VIS=True)
        acc_test = test(models, method, dataloaders, device, mode='test')
        #writer.add_scalars("Loss/train_test", {'train_l':train_loss, 'test_l': test_loss}, epoch)
        #writer.add_scalars("Acc/train_test", {'train_acc':acc_train, 'test_acc': acc_test}, epoch)
        print('epoch: {} \t train_loss: {} \t train_acc: {} \t test_acc: {} \t'.format(epoch, loss, acc_train, acc_test), flush=True)
        if 'backbone' in schedulers :
            schedulers['backbone'].step()
        if 'module' in schedulers and method in ('lloss', 'TA_VAAL'):
            schedulers['module'].step()
        if epoch % 50  == 0 and False:
            acc = test(models, method, dataloaders, device, mode='test')
            if best_acc < acc:
                best_acc = acc
                print('epoch: {} \t Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(epoch, acc, best_acc))
    # writer.flush()
    # writer.close()
    print('>> Finished.')


def log_datamap_confidence(labels, scores, batch_idx, epoch, cfgs):
    # Find confidence for correct label
    confidence = torch.softmax(scores, dim=1)
    confidence_correct_label = [*map(lambda conf, label: conf[label].item(), confidence, labels)]

    # Compute dataset index for samples
    min_idx = batch_idx*cfgs.BATCH
    max_idx = min_idx + len(scores)
    indices = [*range(min_idx, max_idx)]

    # Load pandas dataframe
    results_path = os.path.join(cfgs.results_folder, cfgs.METHOD.replace('-', '_'))
    os.makedirs(results_path, exist_ok=True)
    results = os.path.join(results_path, cfgs.DATASET + '_full.csv')

    # Create a new dataframe if it doesn't exist or if we start a new training
    if not os.path.exists(results) or (epoch == 0 and 0 in indices):
        df = pd.DataFrame(index=[*range(cfgs.OPTIMIZER_BACKBONE.EPOCH_TL+1)],
                          columns=[*range(cfgs.NUM_TRAIN)])
    # Otherwise load it from a file
    else:
        df = pd.read_csv(results, index_col=0)

    # Set confidence values for current epoch and samples
    df.iloc[epoch+1, indices] = confidence_correct_label

    # Set the first row to be the real labels
    if epoch == 0:
        df.iloc[epoch, indices] = labels.tolist()

    df.to_csv(results)
