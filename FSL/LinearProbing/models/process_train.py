
from dataset.balanced_dataset_random_oversampling import LinearProbing_KFold
from models.model import DynMLP2HiddenLit
import pandas as pd
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def extract_features(dataloader):
    features = []
    for batch in dataloader:
        inputs, _ = batch  
        batch_features = [tuple(feature.numpy()) for feature in inputs.view(inputs.size(0), -1)]
        features.extend(batch_features)
    return set(features)  

def run_embed2level_1fold(feature_dir, gt_csv, test_fold, val_fold, dico_level,exp_name,version_name,saving_dir):
    
    """Run one experiment

    Args:
        feature_dir (str or pathlike): path to the directory where the features are stored
        gt_csv (str or pathlike): path to the csv containing the objectifcation annotation
        dico_level (dict): classes to use for the objectification classification
        layers_config (list): list containing the number of neurons for each layer
        exp_name (str): name used to stored the results
        nb_fold (int): number of the fold to use as the validation set
        version_name (str): version_name of the experiment (each validation fold has a different number)
        saving_dir (str or pathlike) : path to the dir where the results should be stored
    """

    train_dataset = LinearProbing_KFold(feature_dir, gt_csv, split="train", test_fold = test_fold, val_fold = val_fold, dico_level =  dico_level, movie_ids=None, auto_split = True)
    val_dataset = LinearProbing_KFold(feature_dir, gt_csv, split="val", test_fold = test_fold, val_fold = val_fold, dico_level =  dico_level, movie_ids=None, auto_split = True)
    test_dataset = LinearProbing_KFold(feature_dir, gt_csv, split="test", test_fold = test_fold, val_fold = val_fold, dico_level = dico_level, movie_ids=None, auto_split = True )

    #added parameter drop_last = True which drop the last batch if it contains fewever example than the batch size
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=19, drop_last=True) 
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset) ,num_workers=19, drop_last=True) #len(val_dataset)
    test_loader= DataLoader(test_dataset, batch_size=len(test_dataset),num_workers=19, drop_last=True) #len(test_dataset)
    
    #check featuer in test and val are not the same
    val_features = extract_features(val_loader)
    test_features = extract_features(test_loader)   
    #check feature intersection
    common_features = val_features.intersection(test_features)
    if common_features:
        print(f"Warning: There are {len(common_features)} overlapping samples in validation and test sets.")
    else:
        print("Validation and test sets are distinct.")

    model = DynMLP2HiddenLit()
    
    es = EarlyStopping(monitor="val_auc", mode="max", patience=10) #PATIENCE CVPR = 5

    ckpt = ModelCheckpoint(
        monitor='val_auc',       
        dirpath='/media/LaCie/SL_Results_TEST/models_ckpts/',    
        filename=f'best-checkpoint-{exp_name}', 
        save_top_k=1, #best model        
        mode='max',            
        save_weights_only=True,  
        verbose=True            
    )   

    logger = TensorBoardLogger(save_dir = saving_dir,name= exp_name, version=version_name)
    trainer = pl.Trainer(min_epochs = 5, max_epochs=100,callbacks=[es, ckpt], logger=logger, check_val_every_n_epoch=1, log_every_n_steps=10,  devices=1) #ORIGINAL CVPR PAPER

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model,test_loader)

def run_embed2level_nfold(feature_dir, gt_csv,dico_level,  exp_name,saving_dir):
    """Run the process for different validation fold (cross-validation)

    Args:
        feature_dir (str or pathlike): path to the directory where the features are stored
        gt_csv (str or pathlike): path to the csv containing the objectifcation annotation
        dico_level (dict): classes to use for the objectification classification
        layers_config (list): list containing the number of neurons for each layer
        exp_name (str): name used to stored the results
        saving_dir (str or pathlike) : path to the dir where the results should be stored
    """

    for test_fold in [0]: #test fold is now the fold 0 [0]
        print("TEST FOLD", test_fold)
        for val_fold in [1]: #val fold is the fold 1 [1]
            if val_fold != test_fold:
                print("VAL_FOLD", val_fold)
                version_name = f'test{test_fold}_val{val_fold}'
                run_embed2level_1fold(feature_dir, gt_csv, test_fold, val_fold, dico_level, exp_name, version_name, saving_dir)
        
        

def process_all_exps(annotation_dir, saving_dir, root_dir):
    
    """Run the classification for several configuration with features extracted from XCLIP retrain on LSMDC

    Args:
        feature_dir (str or pathlike): path to the directory where the features are stored
        annotation_dir (str or pathlike): path to the annotation files
        saving_dir (str or pathlike) : path to the dir where the results should be stored
    """

    root_dir = '/media/LaCie/Features/Neurips_Features/clips_features_max'
    annotation_dir = './dataset/data/'  #directory to the annotation data
    saving_dir = '/media/LaCie/SL_Results_TEST'#directory to save the results

    training_file_names = {"A1_EN_HNS_fold_5.csv": {"Easy Negative": 0, "HNS": 1}  } 
                           
    
    for file_name,dico_level in training_file_names.items():

        gt_file = os.path.join(annotation_dir, file_name)
        exp_name =  file_name.split(".")[0] 

        run_embed2level_nfold(root_dir, gt_file, dico_level,  exp_name, saving_dir)