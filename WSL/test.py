import torch
from sklearn.metrics import precision_score, recall_score,auc, roc_curve, precision_recall_curve, accuracy_score, f1_score, confusion_matrix
import numpy as np
import os


def agg_test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)
        gt = torch.zeros(0)
        total_feat = 0
        for i, input_ in enumerate(dataloader):
            input_, label = input_
            input_ = input_.to(device)

            total_feat += input_.shape[1]
            logits = model(inputs=input_)
      
            logits = torch.squeeze(logits,2) 
            logit_max,_ = torch.max(logits,dim=1)

            sig = logit_max
            pred = torch.cat((pred, sig))
            gt = torch.cat((gt, label))

        
        pred = list(pred.cpu().detach().numpy())


        fpr, tpr, threshold = roc_curve(list(gt), pred)  
        gt_ = [int(g.item()) for g in list(gt)]
        
        pred_ = [int(num >= 0.5) for num in pred]

        
        acc = accuracy_score(y_true=gt_, y_pred=pred_)
        f1 = f1_score(y_true=gt_, y_pred=pred_)
        f1_w = f1_score(y_true=gt_, y_pred=pred_, average='weighted')
        precision_s = precision_score(y_true=gt_, y_pred=pred_)
        recall_s = recall_score(y_true=gt_, y_pred=pred_)
        rec_auc = auc(fpr, tpr)  
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)


        res = {"accuracy": acc,
                "f1_score": f1,
                "f1_score_w": f1_w, 
                "rec_auc":rec_auc, 
                "pr_auc":pr_auc, 

                "precision_score":precision_s, 
                "recall_score":recall_s}
     
        return res
