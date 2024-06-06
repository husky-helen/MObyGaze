import torch
import torch.nn as nn
import torch.nn.init as torch_init



def weight_init_final(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class ModelFinal(nn.Module):
    def __init__(self, n_features, dropout=0.2):
        super(ModelFinal, self).__init__()
              
        
        self.fc1 = nn.Linear(n_features, 256)
        self.fc3 = nn.Linear(256, 1)

        
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.apply(weight_init_final)
 
    def forward(self, inputs):
        bool_len = len(inputs.shape)==3
        if bool_len :
            a,b,c = inputs.shape
            inputs = inputs.view(a*b,c)

        x = self.fc1(inputs)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
   
        x = self.sigmoid(self.fc3(x))
        if bool_len:
            a,b = x.shape
            x = x.view(1,a,b)
        return x
