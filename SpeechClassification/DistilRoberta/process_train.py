import torch
from torch.utils.data import DataLoader
from model.model import DistilRoberta
from dataset.dataset import SpeechDataset
from dataset.dynamic_padding import collate_batch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
#tensorboard
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np 
import os
from pytorch_lightning.callbacks import ModelCheckpoint

def check_label_distribution(loader):
    labels_list = []
    for _, _, labels in loader:
        labels_list.extend(labels.numpy())
    print(f"label distribution: {np.bincount(labels_list)}")
    
def train_model_on_folds(csv_file, train_folds, val_folds, test_folds):
    
    base_name = os.path.basename(csv_file)
    exp_name = os.path.splitext(base_name)[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set cuda 1
    print("DEVICE", device)
    torch.cuda.empty_cache()
        
    all_folds_history = []
        
    for test_fold in test_folds:
        
        test_dataset = SpeechDataset(csv_file, test_fold=test_fold, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last = True, collate_fn=collate_batch, num_workers=19) 
        print("label distribution for test")
        check_label_distribution(test_loader)
        
        for val_fold in val_folds:
            if val_fold == test_fold:
                continue
            
            total_epochs = 40  
            batch_size = 16
            
            t_f = train_folds

            train_dataset = SpeechDataset(csv_file, train_folds=t_f, mode='train')
            val_dataset = SpeechDataset(csv_file, val_fold=val_fold, mode='val')

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = True, collate_fn=collate_batch,  num_workers=19)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, drop_last = True, collate_fn=collate_batch,  num_workers=19) 
            print("label distribution for train")
            check_label_distribution(train_loader)
            print("label distribution for validation")
            check_label_distribution(val_loader)
            
            steps_per_epoch = int(len(train_dataset) / batch_size)  
            total_training_steps = total_epochs * steps_per_epoch
            warmup_steps = int(0.1 * total_training_steps) 
            
            model = DistilRoberta(warmup_steps = warmup_steps).to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_bert_trainable_params = sum(p.numel() for p in model.bert.parameters() if p.requires_grad)
            
            print("PRETRAINED model parameters", total_bert_trainable_params)
            print("Total number of parameters in the entire model: ", total_params)
            print("Number of trainable parameters in the entire model: ", total_params_trainable)
            
            es = EarlyStopping(monitor="val_loss", mode="min", patience=10)
            
            TYPE_OF_MODEL = 'DistillRoberta_FN'
            ckpt = ModelCheckpoint(
                monitor='val_loss',       
                dirpath=f'/Results/{TYPE_OF_MODEL}-ckpts/',    
                filename=f'best-checkpoint-{TYPE_OF_MODEL}-{exp_name}', 
                save_top_k=1, #best model        
                mode='min',            
                save_weights_only=True,  
                verbose=True            
            )   

            logger = TensorBoardLogger(save_dir = '/Results/' ,name= f'{TYPE_OF_MODEL}_{exp_name}', version=f'test_{test_fold}_val_{val_fold}')
            
            trainer = pl.Trainer(min_epochs = 5, max_epochs=total_epochs,callbacks=[es, ckpt], logger=logger, devices=1) #had to reduce the trainer precision (precision parameter) for struct bert because of memory issues # 16-bit precision (model weights get cast to torch.float16)
            
            trainer.fit(model, train_loader, val_loader)
            trainer.test(model, dataloaders=test_loader)

            all_folds_history.append({
                'train': model.trainer.callback_metrics
            })
            
    return all_folds_history

