import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import zipfile
from LinearProbing.models.model import DynMLP2HiddenLit
from dataset.balanced_dataset import LinearProbing_KFold
import pandas as pd

def create_dataloader(features_dir, gt_csv, dico_level, split="test"):
    """Create a DataLoader for the given dataset split."""
    dataset = LinearProbing_KFold(features_dir, gt_csv, split=split, dico_level=dico_level)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=19) 
    return loader

def infer_one_model_one_infertype(model_path, features_dir, gt_csv, dico_level, logger_dir= "/home/xxx/Results/LinearProbing/Inference", base_logger_name = "test", logger_version = "version"):
    """Does the inference for one experiment

    Args:
        model_path (str or pathlike): path to the file containing the checkpoint to load
        features_dir (str or pathlike): path to the directory containing features extracted 
        gt_csv (str or pathlike): path to the objectification annotation file
        dico_level (dict): dictionnary containing the classes to use
        logger_dir (str, optional): Path to the directory where the logs should be saved. Defaults to "/home/xxx/Results/LinearProbing/Inference".
        logger_name (str, optional): name of the exeperiment. Defaults to "test".
        logger_version (str, optional): name of the version of the experiment. Defaults to "version".
        hn_s_experiment_enabled: param for different threshold

    Returns:
        dict: test_results, val_results
    """

    val_loader = create_dataloader(features_dir, gt_csv, dico_level, split="val")
    test_loader = create_dataloader(features_dir, gt_csv, dico_level, split="test")

    model = DynMLP2HiddenLit.load_from_checkpoint(checkpoint_path=model_path, layers_config=None, hn_s_experiment_enabled = True)

    model.eval() 
    
    logger = TensorBoardLogger(save_dir=logger_dir, name=base_logger_name, version=logger_version)
    
    trainer = pl.Trainer(min_epochs = 5, max_epochs=100, logger=logger, check_val_every_n_epoch=1, log_every_n_steps=40, devices=1)  

    with torch.no_grad(): 
        val_results = trainer.validate(model, dataloaders=val_loader)
        test_results = trainer.test(model, dataloaders=test_loader)

    return {'test_results': test_results, 'val_results': val_results}


def run_inference_NeurIPS( print_ = True):
    """Run inference for all the experiments

    Args:
        root_checkpoints (str or pathlike): path to the directory containing the checkpoints
        root_annotations (str or pathlike): path to the objectification annotation directory
        root_logger (str or pathlike): Path to the directory where the logs should be saved.
        features_dir (str or pathlike): path to the directory containing features extracted 
        print_ (bool, optional): If the mean f1 score should be printed. Defaults to True.
    """
    
    root_checkpoints = "/media/LaCie/SL_Results/models_ckpts/"
    root_annotations = '.'
    
    features_dir = '/media/LaCie/Features/Neurips_Features/clips_features_max'
    root_logger = "/media/LaCie/SL_Results/"
    

    root_logg = "/media/LaCie/SL_Results/RES_LOG"
    annotation_dir = './dataset/data/'  #directory to the annotation data


    # CHANGE THE DATASET NAME ACCORDING TO THE EXPERIMENT YOU WANT TO ACHIEVE
    training_file_names = {"A1_EN_HNS_fold_5.csv": {"Easy Negative": 0, "HNS": 1}  }
    
    experiments = os.listdir(root_checkpoints)
    scores_dico = {}
    list_results_test = []
    list_results_val = []
    for experiment,dico_level in training_file_names.items():
        print(experiment)
        scores_dico[experiment] = {}
    
        expname = experiment.split(".")[0]
        annotator, neg, pos, _, numfold =  experiment.split(".")[0].split("_")
        model_path = os.path.join(root_checkpoints, "best-checkpoint-"+ expname+".ckpt")


        
        gt_csv = os.path.join(annotation_dir, experiment ) 
        logger_dir = os.path.join(root_logger, expname)
        logger_name = 'InferOn_' +expname
        logger_version = numfold

        results = infer_one_model_one_infertype(model_path, 
                                    features_dir, 
                                    gt_csv, 
                                    dico_level, 
                                    logger_dir, 
                                    logger_name, 
                                    logger_version)
        print("****************************", expname,"****************************" )
        print("RESULTS : ", results)
        print("\n\n")
        
        for result_type, result in results.items():
            key = f"{expname}_{result_type}"
            if key in scores_dico[experiment]:
                scores_dico[experiment][key] += [result]
            else:
                scores_dico[experiment][key] = [result]
        
        # validation
        dico_res = results['val_results'][0]
        dico_res['subset'] = "validation"
        dico_res["annotator"] = annotator
        dico_res["fold"] = numfold
        dico_res["total_exp_name"] = expname
        dico_res["exp_name"] = neg + "_" + pos
        list_results_test.append(dico_res)

        
        # test 
        dico_res = results['test_results'][0]
        dico_res['subset'] = "test"
        dico_res["annotator"] = annotator
        dico_res["fold"] = numfold
        dico_res["total_exp_name"] = expname
        dico_res["exp_name"] = neg + "_" + pos
        list_results_val.append(dico_res)
            
    df = pd.DataFrame(list_results_test)
    df.to_csv("big_df_test.csv", sep=";")

    df = pd.DataFrame(list_results_val)
    df.to_csv("big_df_val.csv", sep=";")
