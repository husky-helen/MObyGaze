import torch
from torch.utils.data import DataLoader
from model.model import AudioClassifier
from dataset.AudioDataset import AudioDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
#tensorboard
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np 
import os
from pytorch_lightning.callbacks import ModelCheckpoint

def check_label_distribution(loader):
    labels_list = []
    for _, labels in loader:
        labels_list.extend(labels.numpy())
    print(f"label distribution: {np.bincount(labels_list)}")
   
   
def check_dataloader(loader):
    for i, batch in enumerate(loader):
        features, labels = batch
        print(f"Batch {i}: {batch}", f"Features shape: {features.shape}")
        if i >= 5:  #print first 5 batches only
            break
            
def train_model_on_folds(csv_file, base_dir, train_folds, val_folds, test_folds):
    
    base_name = os.path.basename(csv_file)
    exp_name = os.path.splitext(base_name)[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set cuda 1
    print("DEVICE", device)
    torch.cuda.empty_cache()
        
    all_folds_history = []
        
    for test_fold in test_folds:
        
        batch_size = 32 
         
        test_dataset = AudioDataset(csv_file, base_dir, test_fold=test_fold, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last = True, num_workers=19) 
        print("label distribution for test")
        check_label_distribution(test_loader)
        
        print("calculated labels")
        for val_fold in val_folds:
            if val_fold == test_fold:
                continue
            
            total_epochs = 40 
           
            t_f = train_folds 

            train_dataset = AudioDataset(csv_file, base_dir,train_folds=t_f, mode='train')
            val_dataset = AudioDataset(csv_file, base_dir, val_fold=val_fold, mode='val')

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = True,  num_workers=19)
 
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset),  shuffle=False, drop_last = True,num_workers=19) 

            #Check train_loader and val_loader
            print("Checking train_loader...")
            check_dataloader(train_loader)
                
            print("label distribution for train")
            check_label_distribution(train_loader)
            print("label distribution for validation")
            check_label_distribution(val_loader)
          
            steps_per_epoch = int(len(train_dataset) / batch_size)  #depends on the dataset and batch size (total rows / batch size -> around 500 / 16)
            total_training_steps = total_epochs * steps_per_epoch
            warmup_steps = int(0.1 * total_training_steps) 
            
            model = AudioClassifier(warmup_steps = warmup_steps).to(device) 
            
            total_params = sum(p.numel() for p in model.parameters())
        
            print("Total number of parameters in the entire model: ", total_params)
            
            es = EarlyStopping(monitor="val_loss", mode="min", patience=10)
            
            TYPE_OF_MODEL = 'Wav2Vec2'
            ckpt = ModelCheckpoint(
                monitor='val_loss',       
                dirpath=f'/Results/{TYPE_OF_MODEL}-ckpts/',    
                filename=f'{TYPE_OF_MODEL}-{exp_name}', 
                save_top_k=5, #best 10 model        
                mode='min',            
                save_weights_only=True,  
                verbose=True            
            )   

            logger = TensorBoardLogger(save_dir = '/Results/' ,name= f'{TYPE_OF_MODEL}_{exp_name}', version=f'test_{test_fold}_val_{val_fold}')
            
            trainer = pl.Trainer(min_epochs = 5, max_epochs=total_epochs,callbacks=[es, ckpt], logger=logger, devices=1) 
            
            trainer.fit(model, train_loader, val_loader)
            trainer.test(model, dataloaders=test_loader)

            all_folds_history.append({
                'train': model.trainer.callback_metrics
            })
            
    return all_folds_history

