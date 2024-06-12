import torch
from torch import nn
import numpy as np
import transformers
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import pandas as pd
from itertools import cycle
from transformers import BertModel, RobertaModel, DistilBertModel
from sklearn.metrics import auc, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sn
import io
import torchmetrics
from PIL import Image
from torchmetrics.classification import BinaryPrecision
from sklearn.metrics import classification_report
from transformers import AutoModel, AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from sklearn.metrics import auc, precision_recall_curve


class DistilRoberta(pl.LightningModule):
    def __init__(self, num_classes=1, dropout_prob=0.1, warmup_steps=None, training_steps= None): 
        super(DistilRoberta, self).__init__()
        
        
        self.base_learning_rate = 0.00002
        
        self.bert = RobertaModel.from_pretrained('distilroberta-base')
        
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear1 = nn.Linear(768, num_classes)  
        
        self.freezing_all = None 
        
        if self.freezing_all == True:
            print("FREEZE ALL PARAM")
            for param in self.bert.parameters():
                param.requires_grad = False 
        elif self.freezing_all == False: #freeze oly the first 5 layers of the 12 layers bert base model (partly fine tuned)
            print("FINE TUNING BUT FREEZING THE FIRST N LAYERS")
            for i, (name, param) in enumerate(self.bert.named_parameters()):
                if i < 5:
                    param.requires_grad = False
        else: #all trainable parameters
            print("FINE TUNING")
            for param in self.bert.parameters():
                param.requires_grad = True
            
        #criterion
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        #for optimizer:
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        #torch metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary') 
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')
        
        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary' )
        self.test_precision = torchmetrics.Precision(task='binary' )
        
        self.train_recall = torchmetrics.Recall(task='binary' )
        self.val_recall = torchmetrics.Recall(task='binary' )
        self.test_recall = torchmetrics.Recall(task='binary' )
        
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task="binary")
        
        self.val_auc = torchmetrics.AUROC(task='binary') 
        self.test_auc = torchmetrics.AUROC(task='binary')

        
    def forward(self, input_ids, attention_mask=None):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_output = outputs.last_hidden_state[:, 0, :] #standard bert and variations
        cls_output = self.dropout(cls_output)
        logits = self.linear1(cls_output) 
        
        return logits
    
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_masks, labels = batch
        #print("LABELS", labels)
        logits = self(input_ids, attention_masks).squeeze()
        
        loss = self.criterion(logits, labels.float())
        
        probabilities = torch.sigmoid(logits)
        predicted_classes = (probabilities >= 0.5).int()
        

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("train_acc", self.train_accuracy(predicted_classes, labels), on_step=True, on_epoch=True, prog_bar=True, logger=True)
       
        self.log("train_f1", self.train_f1(predicted_classes, labels), on_epoch=True, prog_bar=True)

        self.log("train_precision", self.train_precision(predicted_classes, labels), on_epoch=True, prog_bar=True)
        self.log("train_recall", self.train_recall(predicted_classes, labels), on_epoch=True, prog_bar=True)
        
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        input_ids, attention_masks, labels = batch
        logits = self(input_ids, attention_masks).squeeze()
        loss = self.criterion(logits, labels.float())
        
        probabilities = torch.sigmoid(logits) #in range 0.1 for probabilities
        predicted_classes = (probabilities >= 0.5).int()
        
        labels_np = labels.cpu().numpy()
        preds_np = predicted_classes.cpu().numpy()
        probabilities_np = probabilities.detach().cpu().numpy()
        
        precision_curve, recall_curve, _ = precision_recall_curve(labels_np, probabilities_np)
        pr_auc = auc(recall_curve, precision_curve)
        self.log("val_precision_recall_auc", pr_auc)
        
        print("VALIDATION CLASSIFICATION REPORT")
        print(classification_report(labels_np, preds_np))

        conf_matrix = torchmetrics.functional.confusion_matrix(predicted_classes, labels, task='binary')
        print("Confusion Matrix:\n", conf_matrix.cpu().numpy())

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("val_acc", self.val_accuracy(predicted_classes, labels), on_step=True, on_epoch=True, prog_bar=True, logger=True)
       
        self.log("val_f1", self.val_f1(predicted_classes, labels), on_epoch=True, prog_bar=True)
      
        self.log("val_precision", self.val_precision(predicted_classes, labels), on_epoch=True, prog_bar=True)
        self.log("val_recall", self.val_recall(predicted_classes, labels), on_epoch=True, prog_bar=True)
        
        self.log("val_auc", self.val_auc(probabilities, labels.int()), on_epoch=True, prog_bar=True)
        
        return loss
    
    
    def test_step(self, batch, batch_idx):
        
        input_ids, attention_masks, labels = batch
        logits = self(input_ids, attention_masks).squeeze()
        loss = self.criterion(logits, labels.float())
        
        probabilities = torch.sigmoid(logits) #in range 0.1 for probabilities
        predicted_classes = (probabilities >= 0.5).int()

        labels_np = labels.cpu().numpy()
        preds_np = predicted_classes.cpu().numpy()
        probabilities_np = probabilities.detach().cpu().numpy()
        
        precision_curve, recall_curve, _ = precision_recall_curve(labels_np, probabilities_np)
        pr_auc = auc(recall_curve, precision_curve)
        self.log("test_precision_recall_auc", pr_auc)
        
        print("TEST CLASSIFICATION REPORT")
        print(classification_report(labels_np, preds_np))
        
     
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("test_acc", self.test_accuracy(predicted_classes, labels), on_step=True, on_epoch=True, prog_bar=True, logger=True)
     
        self.log("test_f1", self.test_f1(predicted_classes, labels), on_epoch=True, prog_bar=True)
        
        self.log("test_precision", self.test_precision(predicted_classes, labels), on_epoch=True, prog_bar=True)
        self.log("test_recall", self.test_recall(predicted_classes, labels), on_epoch=True, prog_bar=True)
        
        self.log("test_auc", self.test_auc(probabilities, labels.int()), on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.00002, weight_decay = 0.00005)

        def warmup(current_step: int): 
            if current_step < self.warmup_steps:
                return float(current_step) / float(self.warmup_steps)
            else:
                return max(0.0, float(self.training_steps - current_step) / float(max(1, self.training_steps - self.warmup_steps)))
        
        if self.warmup_steps and self.training_steps: warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)

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
