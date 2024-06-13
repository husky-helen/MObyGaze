from arg_pars import opt
from models.process_train import process_all_exps
from models.process_inference_neurips import run_inference_NeurIPS
from arg_pars import opt
import torch
import random
import numpy as np 
import os

    

def setup_seed(seed=42): 
    random.seed(seed) 
    os.environ["PYTHONHASHSEED"] = str(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) # cpu 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 


if __name__ == '__main__':
    setup_seed(42)
    
    features_dir = opt.features_dir
    annotation_dir = opt.annotation_dir
    training_saving_dir = opt.training_saving_dir
    infering_saving_dir = opt.infering_saving_dir
    root_checkpoints = opt.training_saving_dir

    
    run_inference_NeurIPS() 