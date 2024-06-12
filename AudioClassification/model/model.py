import torch
from torch import nn
import numpy as np
import transformers
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from sklearn.metrics import auc
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import seaborn as sn
import io
import torchmetrics
from PIL import Image
from torchmetrics.classification import BinaryPrecision
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, classification_report, confusion_matrix, auc, roc_curve

class AudioClassifier(pl.LightningModule):
    def __init__(self, input_dim=1024, num_classes=1, dropout_prob=0.1, warmup_steps=None): 
        super(AudioClassifier, self).__init__()
        
        self.base_learning_rate = 0.0005 
        
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(input_dim, 1)
        #MLP
        #self.linear = nn.Linear(input_dim, 256) 
        #self.bn1 = nn.BatchNorm1d(256)
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=dropout_prob)    
        #self.linear2 = nn.Linear(256, num_classes)
        
        #criterion
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        #for optimizer:
        self.warmup_steps = warmup_steps

    def forward(self, input_ids):
        
        input_ids = self.dropout(input_ids)
        
        x = self.linear(input_ids)
        #x = self.bn1(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        #x = self.linear2(x)

        
        return x
    
    
    def training_step(self, batch, batch_idx):
        
        input_ids, labels = batch
        
        logits = self(input_ids)
        labels = labels.unsqueeze(1)
        
        loss = self.criterion(logits, labels.float())
        
        probabilities = torch.sigmoid(logits)
        predicted_classes = (probabilities >= 0.5).int()
        
        labels_np = labels.cpu().numpy().astype(int)
        preds_np = predicted_classes.cpu().numpy()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        
        train_acc = accuracy_score(labels_np, preds_np)
        precision = precision_score(labels_np, preds_np, zero_division=0)
        recall = recall_score(labels_np, preds_np, zero_division=0)
        
        train_f1 = f1_score(labels_np, preds_np, zero_division=0)
        train_f1_weighted = f1_score(labels_np, preds_np, average='weighted', zero_division=0)
        
        self.log("train_acc", train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1", train_f1, on_epoch=True, prog_bar=True)
        self.log("train_f1_weighted", train_f1_weighted, on_epoch=True, prog_bar=True)
        self.log("train_precision", precision, on_epoch=True, prog_bar=True)
        self.log("train_recall", recall, on_epoch=True, prog_bar=True)

        return loss
    
    @staticmethod
    def replace_nan_with_minus_one(arr):
        arr[np.isnan(arr)] = -1
        return arr

    def validation_step(self, batch, batch_idx):

        input_ids, labels = batch
        
        logits = self(input_ids)

        labels = labels.unsqueeze(1)
        
        loss = self.criterion(logits, labels.float())
        
        probabilities = torch.sigmoid(logits) 
    
        predicted_classes = (probabilities >= 0.5).int()
        
        labels_np = labels.cpu().numpy().astype(int)
        preds_np = predicted_classes.cpu().numpy()
        probabilities_np = probabilities.detach().cpu().numpy()
        
        print("VALIDATION CLASSIFICATION REPORT")
        print(classification_report(labels_np, preds_np))

        conf_matrix = torchmetrics.functional.confusion_matrix(predicted_classes, labels.int(), task='binary')
        print("Confusion Matrix:\n", conf_matrix.cpu().numpy())
        
        acc = accuracy_score(labels_np, preds_np)
        f1 = f1_score(labels_np, preds_np, zero_division=0)
        f1_w = f1_score(labels_np, preds_np, average='weighted', zero_division=0) 
        precision = precision_score(labels_np, preds_np, zero_division=0)

        recall = recall_score(labels_np, preds_np, zero_division=0)

        fpr, tpr, threshold = roc_curve(labels_np, probabilities_np) 
        
        fpr = self.replace_nan_with_minus_one(fpr)
        tpr = self.replace_nan_with_minus_one(tpr)
        
        rec_auc = auc(fpr, tpr) if len(fpr) > 1 and len(tpr) > 1 else -1 
        
        
        precision_curve, recall_curve, _ = precision_recall_curve(labels_np, probabilities_np)
        
        precision_curve = self.replace_nan_with_minus_one(precision_curve)
        recall_curve = self.replace_nan_with_minus_one(recall_curve)
        pr_auc = auc(recall_curve, precision_curve) if len(precision_curve) > 1 and len(recall_curve) > 1 else -1
       
        self.log('val_loss', loss)
        self.log("val_acc", acc, on_step=True, on_epoch=True)
        self.log('val_f1', f1)
        self.log('val_f1_weighted', f1_w)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_auc", rec_auc)
        self.log("val_pr_auc", pr_auc)

        
        return loss
    
    
    def test_step(self, batch, batch_idx):
        
        input_ids, labels = batch
              
        logits = self(input_ids)
        labels = labels.unsqueeze(1)

        loss = self.criterion(logits, labels.float())
        
        probabilities = torch.sigmoid(logits) 
        predicted_classes = (probabilities >= 0.5).int()

        labels_np = labels.cpu().numpy().astype(int)
        preds_np = predicted_classes.cpu().numpy()
        probabilities_np = probabilities.detach().cpu().numpy()
        
        print("TEST CLASSIFICATION REPORT")
        print(classification_report(labels_np, preds_np))
        
        
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
  
        acc = accuracy_score(labels_np, preds_np)
        f1 = f1_score(labels_np, preds_np, zero_division=0)
        f1_w = f1_score(labels_np, preds_np, average='weighted', zero_division=0)
        precision = precision_score(labels_np, preds_np, zero_division=0)
        recall = recall_score(labels_np, preds_np, zero_division=0)

        fpr, tpr, _ = roc_curve(labels_np, probabilities_np)

        fpr = self.replace_nan_with_minus_one(fpr)
        tpr = self.replace_nan_with_minus_one(tpr)
        
        rec_auc = auc(fpr, tpr) if len(fpr) > 1 and len(tpr) > 1 else -1 
        
        
        precision_curve, recall_curve, _ = precision_recall_curve(labels_np, probabilities_np)
        
        precision_curve = self.replace_nan_with_minus_one(precision_curve)
        recall_curve = self.replace_nan_with_minus_one(recall_curve)
        pr_auc = auc(recall_curve, precision_curve) if len(precision_curve) > 1 and len(recall_curve) > 1 else -1
        
        self.log("test_acc", acc)
        self.log('test_f1', f1)
        self.log('test_f1_weighted', f1_w)
        self.log("test_precision", precision)
        self.log("test_recall", recall)

        self.log("test_auc", rec_auc)
        self.log("test_pr_auc", pr_auc)


        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.base_learning_rate, weight_decay = 0.00005)  
        
        reduce_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True) 
        
        return {"optimizer": optimizer, "lr_scheduler": reduce_scheduler, "monitor": "val_loss"} 
    
    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_closure, 
        on_tpu=False, using_native_amp=False, using_lbfgs=False):

        optimizer.step(closure=optimizer_closure)
       
        if self.trainer.global_step < self.warmup_steps:
            
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.base_learning_rate 
                
        optimizer.zero_grad()


