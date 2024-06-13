import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
from sklearn.metrics import auc, roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import io
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as torch_init



def weight_init(m, He = True): # He weight initialisation
    classname = m.__class__.__name__

    if He: 
        if classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0)
    else:
        #xavier uniform
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            torch_init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
                

class DynMLP2HiddenLit(pl.LightningModule):

    def __init__(self,layers_config = None, w = None):
       
        """Args:
            layers_config (list): _description_
            activation (torch function, optional): Activation function to use. Defaults to nn.ReLU().
            w (torch.tensor, optional): weights to use in the loss. Defaults to None.
            average (str) : average to use in the computation of the metrics (f1 score, recall, precision)
        """
        super(DynMLP2HiddenLit, self).__init__()
        self.val_total_positive = 0
        self.val_total_negative = 0
        
        self.test_total_positive = 0
        self.test_total_negative = 0
        
        self.fc1 = nn.Linear(512, 256) 
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, 1)

        self.loss_ce = nn.BCEWithLogitsLoss(weight=w)
        self.apply(weight_init)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        binary_targets = targets[:, 1].unsqueeze(1)
        logits = self(inputs)

        loss = self.loss_ce(logits, binary_targets) #outputs logits 

        probabilities = torch.sigmoid(logits)
        
        pred_label = (probabilities >= 0.5).int()
        
        binary_targets_np = binary_targets.detach().cpu().numpy()
        pred_label_np = pred_label.detach().cpu().numpy()
        
        train_acc = accuracy_score(binary_targets_np, pred_label_np)
        precision = precision_score(binary_targets_np, pred_label_np, zero_division=0)
        recall = recall_score(binary_targets_np, pred_label_np, zero_division=0)
        
        train_f1 = f1_score(binary_targets_np, pred_label_np, zero_division=0)
        train_f1_weighted = f1_score(binary_targets_np, pred_label_np, average='weighted', zero_division=0)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1", train_f1, on_epoch=True, prog_bar=True)
        self.log("train_f1_weighted", train_f1_weighted, on_epoch=True, prog_bar=True)
        self.log("train_precision", precision, on_epoch=True, prog_bar=True)
        self.log("train_recall", recall, on_epoch=True, prog_bar=True)

     
        return loss 
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        binary_targets = targets[:, 1].unsqueeze(1)
        
        self.val_total_positive += torch.sum(binary_targets == 1).item()
        self.val_total_negative += torch.sum(binary_targets == 0).item()
        
        
        logits = self(inputs)
        
        loss = self.loss_ce(logits, binary_targets)  #loss function expects logits then applies the inner sigmoid layer

        probabilities = torch.sigmoid(logits)
        
        pred_label = (probabilities >= 0.5).int()  
        
        binary_targets_np = binary_targets.detach().cpu().numpy().astype(int)
        pred_label_np = pred_label.detach().cpu().numpy()
        probabilities_np = probabilities.detach().cpu().numpy()
        
        
        
        #calculate metrics using binary predictions based on probabilities
        acc = accuracy_score(binary_targets_np, pred_label_np)
        f1 = f1_score(binary_targets_np, pred_label_np, zero_division=0)
        f1_w = f1_score(binary_targets_np, pred_label_np, average='weighted', zero_division=0) 
        precision = precision_score(binary_targets_np, pred_label_np, zero_division=0)

        recall = recall_score(binary_targets_np, pred_label_np, zero_division=0)

        fpr, tpr, threshold = roc_curve(binary_targets_np, probabilities_np) 
        rec_auc = auc(fpr, tpr)
        
        precision_curve, recall_curve, _ = precision_recall_curve(binary_targets_np, probabilities_np)
        pr_auc = auc(recall_curve, precision_curve)
    
        self.log_conf_matrix(binary_targets.cpu(), pred_label.cpu(), "Val Confusion Matrix")
        self.log('val_loss', loss)
        self.log("val_acc", acc, on_step=True, on_epoch=True)
        self.log('val_f1', f1)
        self.log('val_f1_weighted', f1_w)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_auc", rec_auc)
        self.log("val_pr_auc", pr_auc)
        
       
            
        return loss
    
    def on_validation_epoch_end(self):
        self.log("val_total_positive", self.val_total_positive, on_epoch=True, prog_bar=True)
        self.log("val_total_negative", self.val_total_negative, on_epoch=True, prog_bar=True)

        print(f"Validation - Total Positive: {self.val_total_positive}, Total Negative: {self.val_total_negative}")

        self.val_total_positive = 0
        self.val_total_negative = 0
    
    def log_conf_matrix(self, target_classes, pred_label, title='Confusion Matrix'):
        """Function to be able to load a plot of the confusion matrix in tensorboard

        Args:
            target_classes (torch.tensor): Ground truth classes
            pred_label (torch.tensor): Predicted classes
            title (str, optional): Title of the plot. Defaults to 'Confusion Matrix'.
        """
        confusion = confusion_matrix(target_classes, pred_label)
        fig = plt.figure()
        sn.heatmap(confusion, annot=True, fmt='g')  
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix')

        # Save the figure as an image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0) 
        pil_img = Image.open(buf)

        transform = transforms.ToTensor()
        tensor_img = transform(pil_img)


        #Load image in tensorboard
        self.logger.experiment.add_image(title, tensor_img, global_step=self.global_step)

        buf.close()
        plt.close()

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        binary_targets = targets[:, 1].unsqueeze(1)
        
        self.test_total_positive += torch.sum(binary_targets == 1).item()
        self.test_total_negative += torch.sum(binary_targets == 0).item()
        
        logits = self(inputs)

        loss = self.loss_ce(logits, binary_targets) 
        
        probabilities = torch.sigmoid(logits)
        
        pred_label = (probabilities >= 0.5).int()

        binary_targets_np = binary_targets.detach().cpu().numpy().astype(int)
        pred_label_np = pred_label.detach().cpu().numpy()
        probabilities_np = probabilities.detach().cpu().numpy()
        
        acc = accuracy_score(binary_targets_np, pred_label_np)
        f1 = f1_score(binary_targets_np, pred_label_np, zero_division=0)
        f1_w = f1_score(binary_targets_np, pred_label_np, average='weighted', zero_division=0)
        precision = precision_score(binary_targets_np, pred_label_np, zero_division=0)
        recall = recall_score(binary_targets_np, pred_label_np, zero_division=0)
    
        fpr, tpr, _ = roc_curve(binary_targets_np, probabilities_np)
        rec_auc = auc(fpr, tpr)
        precision_curve, recall_curve, _ = precision_recall_curve(binary_targets_np, probabilities_np)
        pr_auc = auc(recall_curve, precision_curve)
        
        self.log("test_acc", acc)
        self.log('test_f1', f1)
        self.log('test_f1_weighted', f1_w)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        
        self.log("test_auc", rec_auc)
        self.log("test_pr_auc", pr_auc)

            
        
        return loss
    
    def on_test_epoch_end(self):
        print(f"Test - Total Positive: {self.test_total_positive}, Total Negative: {self.test_total_negative}")
        self.log("test_total_positive", self.test_total_positive, on_epoch=True, prog_bar=True)
        self.log("test_total_negative", self.test_total_negative, on_epoch=True, prog_bar=True)

        self.test_total_positive = 0
        self.test_total_negative = 0


    def configure_optimizers(self):
        """When using elasticnet set the param weigh_decay = 0 as uit combines l1 + l2 regularisation techniques.
        
        (l2 regularisation -> weight decay > 0)

        Returns:
            _type_: _description_
        """
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay = 0.00005) #weight_decay 0.00005 = l2 regularisation
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.1, verbose=True) #set mode to max as with auc want to maximise
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_auc"}
