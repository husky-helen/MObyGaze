from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def BMS(prediction, ground_truths):
    """Compute the metrics in a winner takes it all fashion

    Args:
        prediction (torch.tensor): Tensor of size (nb_data), stores the predictions
        ground_truths (torch.tensor): Tensor of size (nb_data, nb_label), ground_truths[i,j] stores the weight for the data number i and label j
    
    Returns: dict containing the values of the metrics
    """

    nb_data, nb_annot = ground_truths.shape

    GT_tensor = torch.repeat_interleave(torch.arange(nb_annot).reshape(1,-1), nb_data, axis=0).to(prediction.device)
    predictions = torch.repeat_interleave(prediction.reshape(-1,1), nb_annot, axis = 1).to(prediction.device)
    dist = torch.abs(predictions - GT_tensor)

    distance = dist.masked_fill(ground_truths == 0, 10000)
    GT_label = torch.argmin(distance, dim = 1).cpu().to(prediction.dtype)
    prediction = torch.squeeze(prediction.cpu())

    l = F.binary_cross_entropy(prediction, GT_label)
    pred_label = (prediction>0.5).int()


    acc = accuracy_score(GT_label, pred_label.cpu())
    f1 = f1_score(GT_label, pred_label.cpu())
    f1w = f1_score(GT_label, pred_label.cpu(), average='weighted')
    recall = recall_score(GT_label, pred_label.cpu())
    precision = precision_score(GT_label, pred_label.cpu(), zero_division=0)


    roc_auc = roc_auc_score(GT_label, prediction.cpu())   
    p,r, th = precision_recall_curve(GT_label, prediction)

    auc_pr = auc(r,p)
 

    result_dico = {"bms_accuracy": acc, 
                   "bms_f1":f1, 
                   "bms_f1_w":f1w,
                   "bsm_recall": recall, 
                   "bms_precision":precision, 
                   "bms_rocauc":roc_auc,
                   "bms_prauc":auc_pr,
                   "bms_loss": l.item()}
    
    return result_dico

