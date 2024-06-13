
import process_train_llama
import torch 
import os
import numpy as np 
import random

if torch.cuda.is_available():
    print("CUDA is available")
    print("Number of CUDA devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available")
    
    
def setup_seed(seed=42): 
    random.seed(seed) 
    os.environ["PYTHONHASHSEED"] = str(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) # cpu 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    
def main(data_path, train_folds, val_folds, test_folds):
    process_train_llama.train_model_on_folds(data_path, train_folds, val_folds, test_folds)

if __name__ == '__main__':
    setup_seed(42)
    
    
    train_folds = [2]
    val_folds = [1]
    test_folds = [0]
    
    annotators = ['annotator_1', 'annotator_2']

  
    for annotator in annotators:
        
        data_path = [f'/data/Folds/{annotator}_speech_fold_1.csv',
                    f'/data/Folds/{annotator}_speech_fold_2.csv',
                    f'/data/Folds/{annotator}_speech_fold_3.csv',
                    f'/data/Folds/{annotator}_speech_fold_4.csv',
                    f'/data/Folds/{annotator}_speech_fold_5.csv'              
                    ]
        
        for el in data_path:
            main(el, train_folds, val_folds, test_folds)


    