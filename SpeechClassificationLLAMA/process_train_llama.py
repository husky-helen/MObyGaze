import torch
from torch.utils.data import DataLoader
from model.model_llama import LLamaForSequenceClf
from dataset.dataset import SpeechDataset
from dataset.dynamic_padding_llama import collate_batch
import pytorch_lightning as pl
#tensorboard
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np 
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import os 

def check_label_distribution(loader):
    labels_list = []
    for ids, attention, labels in loader:
        labels_list.extend(labels.numpy())
    print(f"label distribution: {np.bincount(labels_list)}")
    
def train_model_on_folds(csv_file, train_folds, val_folds, test_folds):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set cuda 1
    print("DEVICE", device)
    torch.cuda.empty_cache()
    
    base_name = os.path.basename(csv_file)
    exp_name = os.path.splitext(base_name)[0]

    TYPE_OF_MODEL = 'LLAMA-7b'
                   
    all_folds_history = []
        
    for test_fold in test_folds:
        
        test_dataset = SpeechDataset(csv_file, test_fold=test_fold, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last = True, collate_fn=collate_batch, num_workers=19) #DISCUSS THAT WITH JULIE (AS IN PROCESS_TRAIN)
        print("label distribution for test")
        check_label_distribution(test_loader)
        
        for val_fold in val_folds: 
            if val_fold == test_fold:
                continue
            
            total_epochs = 40
            batch_size = 1
            
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
            
            model = LLamaForSequenceClf(warmup_steps = warmup_steps).to(device)
            
            es = EarlyStopping(monitor="val_loss", mode="min", patience=10)
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
              
            logger = TensorBoardLogger(save_dir = '/results/SpeechDetectionLLAMA' ,name= f'{TYPE_OF_MODEL}_{exp_name}', version=f'test_{test_fold}_val_{val_fold}')
            
            ckpt = ModelCheckpoint(
                monitor='val_loss',       
                dirpath=f'/results/SpeechDetectionLLAMA/{TYPE_OF_MODEL}-ckpts/',    
                filename=f'best-checkpoint-{TYPE_OF_MODEL}-{exp_name}', 
                save_top_k=1, #best model        
                mode='min',            
                save_weights_only=True,  
                verbose=True            
            )   
            
            trainer = pl.Trainer(min_epochs = 1, max_epochs=total_epochs,
                                 accumulate_grad_batches = 8, #Gradient accumulation to simulate a bigger batch size 
                                 callbacks=[es, lr_monitor, ckpt], 
                                 logger=logger, devices=1)
            
            trainer.fit(model, train_loader, val_loader) 
            trainer.test(model, dataloaders=test_loader)

            all_folds_history.append({
                'train': model.trainer.callback_metrics
            })
            
    return all_folds_history

