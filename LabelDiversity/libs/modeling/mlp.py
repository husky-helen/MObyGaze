import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from .models import register_backbone



def weight_init_final(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    
       
@register_backbone("final_binary")
class ModelFinalBinary(nn.Module):
    def __init__(self, n_features, dropout=0.2, loss_function = F.binary_cross_entropy):
        super(ModelFinalBinary, self).__init__()
              
        
        self.fc1 = nn.Linear(n_features, 256)
        self.fc3 = nn.Linear(256, 1)

        
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.apply(weight_init_final)

        self.loss_function = loss_function
 
    def forward(self, inputs):
        feats, labels = inputs


        bool_len = len(feats.shape)==3
        if bool_len :
            a,b,c = feats.shape
            inputs = feats.view(a*b,c)

        x = self.fc1(feats)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        
        x = self.sigmoid(self.fc3(x))
   
        if bool_len:
            a,b = x.shape
            x = x.view(1,a,b)

        loss = self.loss(x, labels)

        return x, loss

    def loss(self, preds, ground_truth):
        if len(preds.shape)>1: p = preds.reshape(-1)
        else: p = preds[0]

        return {'final_loss':self.loss_function(p, ground_truth.to(torch.float32))}