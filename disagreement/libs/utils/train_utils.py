import os
import shutil
import time
import pickle

import numpy as np
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from libs.utils.metric_utils import BMS
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, roc_auc_score, auc, precision_recall_curve



################################################################################
def fix_random_seed(seed, include_cuda=True):
    """Args:
    seed (int): number of the seed to use
    include_cude (bool): init seed for torch 
    
    Fix random seed
    
    Returns: the torch manual seed"""

    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder, file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    """prints model parameters

    Args:
        model (torch.nn.Module): torch model to print
    """
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return



class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



################################################################################
def train_one_epoch(train_loader,model,optimizer,scheduler,curr_epoch,tb_writer = None, print_freq = 20):
    """Training function for one epoch

    Args:
        train_loader (torch Dataloader): Dataloader used for training
        model (torch.nn.Module): Model to train
        optimizer (torch optimize): Optimizer to use
        scheduler (torch_scheduler): Scheduler to use
        curr_epoch (int): Number of the current epoch
        tb_writer (tensorboard logger, optional): Tensorboard logger. Defaults to None.
        print_freq (int, optional): Printing frequency. Defaults to 20.
    """   
   
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()
    losses = {}
    # main training loop
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # zero out optim
        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model

        proba, losses = model(video_list)
        losses['final_loss'].backward()
     
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step(metrics = losses["final_loss"])


        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars(
                    'train/all_losses',
                    tag_dict,
                    global_step
                )
                # final loss
                tb_writer.add_scalar(
                    'train/final_loss',
                    losses_tracker['final_loss'].val,
                    global_step
                )

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4  += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )

            print('\t'.join([block1, block2, block3, block4]))
    tb_writer.add_scalar(
                'train/final_loss_epoch',
                losses_tracker['final_loss'].avg,
                curr_epoch
                )

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return


def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    output_file = None,
    tb_writer = None,
    print_freq = 20,
    dir_writer='validation'
):
    """Validation function for one epoch

    Args:
        val_loader (torch Dataloader): _description_
        model (torch.nn.Module): _description_
        curr_epoch (int): Number of the current epoch
        output_file (_type_, optional): _description_. Defaults to None.
        tb_writer (tensorboard logger, optional): Tensorboard logger. Defaults to None.
        print_freq (int, optional): Printing frequency. Defaults to 20.
        dir_writer (str, optional): _description_. Defaults to 'validation'.

    Returns:
        dict: contains the metrics results
    """
    # either evaluate the results or save the results
    assert output_file is not None

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    
    losses_tracker = {}
    losses = {}
  
    start = time.time()
    Probas = []
    Labels = []

    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            feats, labels = video_list
            # print("feats : ", feats.shape)
            proba, losses = model(video_list)
            # print("PROBABILITIES IN VALID ON EPOCH SOFT : ",proba)
            Probas.append(proba)
            Labels.append(labels)
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                #print("Dans le train : , loss tracker update")
                losses_tracker[key].update(value.item())
                

                # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))
    
    
    probas = torch.concat(Probas)
    targets = torch.concat(Labels)
    
    pred_label = (probas>0.5).int()

    target_classes = targets
    acc = accuracy_score(target_classes.cpu(), pred_label.cpu())
    f1 = f1_score(target_classes.cpu(), pred_label.cpu())
    f1w = f1_score(target_classes.cpu(), pred_label.cpu(), average='weighted')
    recall = recall_score(target_classes.cpu(), pred_label.cpu())
    precision = precision_score(target_classes.cpu(), pred_label.cpu(), zero_division=0)

    
    roc_auc = roc_auc_score(target_classes.cpu(), probas.cpu())   
    p,r, th = precision_recall_curve(target_classes.cpu(), probas.cpu())
 
    aucpr = auc(r,p)

    
    tb_writer.add_scalar(dir_writer + '/final_loss_epoch',losses_tracker['final_loss'].avg,curr_epoch)
    tb_writer.add_scalar(dir_writer+'/acc', acc, curr_epoch)
    tb_writer.add_scalar(dir_writer+'/f1', f1, curr_epoch)
    tb_writer.add_scalar(dir_writer+'/f1_w', f1w, curr_epoch)
    tb_writer.add_scalar(dir_writer+'/recall', recall, curr_epoch)
    tb_writer.add_scalar(dir_writer+'/precision', precision, curr_epoch)
    tb_writer.add_scalar(dir_writer+'/roc_auc', roc_auc, curr_epoch)
    tb_writer.add_scalar(dir_writer+'/pr_auc', aucpr, curr_epoch)

    result_dico = {"accuracy": acc, "f1":f1, "f1_w":f1w,"recall": recall, "precision":precision, "roc_auc":roc_auc, "pr_auc":aucpr, "loss": losses_tracker['final_loss'].avg}
    return losses, result_dico



def valid_one_epoch_bms(
    val_loader,
    model,
    curr_epoch,

    output_file = None,
    tb_writer = None,
    print_freq = 20,
    dir_writer='validation'
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert output_file is not None

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    
    losses_tracker = {}
    losses = {}
    # loop over validation set
    start = time.time()
    Probas = []
    Labels = torch.zeros((len(val_loader),2))


    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            feats, labels = video_list
            Labels[iter_idx, :] = labels
            # print("feats : ", feats.shape)
            proba, losses = model(video_list)
            Probas.append(proba)
            
            # Labels.append(hard_label)
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                #print("Dans le train : , loss tracker update")
                losses_tracker[key].update(value.item())
                

                # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            #print("Dans le train batch_time update à la fin")
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))
    
    
    probas = torch.concat(Probas)
    targets = Labels.to(probas.device)

    
    result_dico_bms = BMS(prediction = probas, ground_truths = targets)
        
    tb_writer.add_scalar(
                dir_writer + '/final_loss_epoch',
                losses_tracker['final_loss'].avg,
                curr_epoch
                )
    
    acc = result_dico_bms["bms_accuracy"]
    f1 = result_dico_bms["bms_f1"]
    f1w = result_dico_bms["bms_f1_w"]
    recall = result_dico_bms["bsm_recall"]
    precision = result_dico_bms["bms_precision"]
    auc_roc = result_dico_bms["bms_rocauc"]
    l_ = result_dico_bms["bms_loss"]
    
    tb_writer.add_scalar(dir_writer+'/bms_acc', acc, curr_epoch)
    tb_writer.add_scalar(dir_writer+'/bms_f1', f1, curr_epoch)
    tb_writer.add_scalar(dir_writer+'/bms_f1_w', f1w, curr_epoch)
    tb_writer.add_scalar(dir_writer+'/bms_recall', recall, curr_epoch)
    tb_writer.add_scalar(dir_writer+'/bms_precision', precision, curr_epoch)
    tb_writer.add_scalar(dir_writer+'/bms_auc_roc', auc_roc, curr_epoch)
    tb_writer.add_scalar(dir_writer+'/bms_loss', l_, curr_epoch)
    

    return losses, result_dico_bms