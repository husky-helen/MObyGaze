import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def CE(logits, ground_truth, w=None):
    return F.cross_entropy(logits, ground_truth, weight=w)

@torch.jit.script
def binary_CE(logits, ground_truth, w=None):
    proba = torch.sigmoid(logits)
    return F.binary_cross_entropy(proba, ground_truth, weight=w)

@torch.jit.script
def MSE(preds, ground_truth):
    return F.mse_loss(preds, ground_truth)

@torch.jit.script
def soft_loss(preds, ground_truth):
    """
    Computes the mean of the losses for each annotator labels
    preds should be the output of a sigmoid
    ground_truth shape is batch_size x nb_classes """

    l = torch.tensor([0], device = preds.device, dtype = ground_truth.dtype)
    bs = ground_truth.shape[0]
    for i in range(ground_truth.shape[1]):
        tmp_gt = torch.ones(bs).to(preds.device) * i
        tmp_bce =  F.binary_cross_entropy(preds, tmp_gt, reduction='none')
        tmp_wbce = ground_truth[:,i] * tmp_bce

        l[0] += torch.sum(tmp_wbce)
    
    total_w = torch.sum(ground_truth)

    return l / total_w

def inverted_variety_loss(preds, ground_truth):
    """Take the gt the closest to the prediction

    Args:
        preds (torch.tensor): prediction of the model
        ground_truth (torch.tensor): ground truth labels
    """
    bs, nb_label = ground_truth.shape
    GT_tensor = torch.arange(nb_label, device = preds.device).reshape(1, nb_label)

    GT_tensor = GT_tensor.repeat(preds.shape[0], 1)
    preds_ = preds.reshape(-1,1).repeat(1, nb_label)

    distance = torch.abs(preds_ - GT_tensor)
    distance = distance.masked_fill(ground_truth == 0, 10000)
    GT_label = torch.argmin(distance, dim = 1).to(preds.dtype)

    l = F.binary_cross_entropy(preds, GT_label)  
    return l