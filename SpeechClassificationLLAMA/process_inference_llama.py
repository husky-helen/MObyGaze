import torch
from torch.utils.data import DataLoader
from model.model_llama import LLamaForSequenceClf
from dataset.dataset import SpeechDataset
from dataset.dynamic_padding_llama import collate_batch
import pytorch_lightning as pl
#tensorboard
import numpy as np 
import os 
from sklearn.metrics import auc, roc_curve, recall_score, precision_score, precision_recall_curve, accuracy_score, f1_score

def check_label_distribution(loader):
    labels_list = []
    for ids, attention, labels in loader:
        labels_list.extend(labels.numpy())
    print(f"label distribution: {np.bincount(labels_list)}")
    
def infer_model_on_folds(csv_file, train_folds, val_folds, test_folds, model_path):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set cuda 1
    print("DEVICE", device)
    torch.cuda.empty_cache()
    
    base_name = os.path.basename(csv_file)
    exp_name = os.path.splitext(base_name)[0]

    TYPE_OF_MODEL = 'LLAMA-7b'
        
    all_folds_history = []
        
    for test_fold in test_folds:
        
        test_dataset = SpeechDataset(csv_file, test_fold=test_fold, mode='test')

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last = True, collate_fn=collate_batch, num_workers=19) 
        print("label distribution for test")
        check_label_distribution(test_loader)
        
        t_f = train_folds 

        model = LLamaForSequenceClf.load_from_checkpoint(checkpoint_path=model_path).to(device)
            
        all_preds = []

        all_targets= []
        for batch in test_loader:
            inputs_ids, attention_masks, labels = batch
            inputs_ids, attention_masks, y = inputs_ids.to(device), attention_masks.to(device), labels.to(device)
       
            with torch.no_grad():
                y_hat = model(inputs_ids, attention_masks)
                y_hat = torch.sigmoid(y_hat)
                all_preds.append(y_hat.cpu().numpy())
                all_targets.append(y.cpu().numpy())
    

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    pred_labe = [int(num >= 0.5) for num in all_preds]
    
    acc = accuracy_score(y_true=all_targets, y_pred=pred_labe)
    f1 = f1_score(y_true=all_targets, y_pred=pred_labe)
    f1_w = f1_score(y_true=all_targets, y_pred=pred_labe, average='weighted')
    
    fpr, tpr, threshold = roc_curve(all_targets, all_preds)
    rec_auc = auc(fpr, tpr)

    precision, recall, th = precision_recall_curve(all_targets, all_preds)
    pr_auc = auc(recall, precision)

    recall = recall_score(all_targets, pred_labe)
    precision = precision_score(all_targets, pred_labe, zero_division=0)

    print("TEST METRICS", csv_file)
    print("Accuracy : ", acc)
    print("F1 : ", f1)
    print("F1W : ", f1_w)
    print("AUC ROC : ", rec_auc)
    print("AUC PR : ", pr_auc)
    print("Precision : ", precision)
    print("Recall : ", recall)
