#import process_train
import process_inference_llama
import torch 
import os
import numpy as np 
import random
import argparse
#torch.cuda.empty_cache()

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


def main(data_path, train_folds, val_folds, test_folds, model_path):    
    process_inference_llama.infer_model_on_folds(data_path, train_folds, val_folds, test_folds, model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process an integer.')

    parser.add_argument('number', type=int, help='An integer number')
    parser.add_argument('exp', type=str, help='finetuning or linearprobing')

    
    args = parser.parse_args()

    print("INFERENCE ")
    setup_seed(42)

    dico_models = {"finetuning" :  {"annotator_1_speech_fold_1": "best-checkpoint-LLAMA-7b-annotator_1_speech_fold_1-v2.ckpt",
"annotator_1_speech_fold_2":"best-checkpoint-LLAMA-7b-annotator_1_speech_fold_2.ckpt",
"annotator_1_speech_fold_3":"best-checkpoint-LLAMA-7b-annotator_1_speech_fold_3.ckpt",
"annotator_1_speech_fold_4":"best-checkpoint-LLAMA-7b-annotator_1_speech_fold_4.ckpt",
"annotator_1_speech_fold_5":"best-checkpoint-LLAMA-7b-annotator_1_speech_fold_5.ckpt",
"annotator_2_speech_fold_1":"best-checkpoint-LLAMA-7b-annotator_2_speech_fold_1.ckpt",
"annotator_2_speech_fold_2":"best-checkpoint-LLAMA-7b-annotator_2_speech_fold_2.ckpt",
"annotator_2_speech_fold_3":"best-checkpoint-LLAMA-7b-annotator_2_speech_fold_3.ckpt",
"annotator_2_speech_fold_4":"best-checkpoint-LLAMA-7b-annotator_2_speech_fold_4.ckpt",
"annotator_2_speech_fold_5":"best-checkpoint-LLAMA-7b-annotator_2_speech_fold_5.ckpt"} }
    
    root_dir = "/home/user/Folds/"
    model_root = "/home/user/results/SpeechDetectionLLAMA/LLAMA-7b-ckpts"
    
    data_path = os.listdir(root_dir)

    train_folds = [2]
    val_folds = [1]
    test_folds = [0]

    el = os.path.join(root_dir, data_path[args.number])

    model_path = os.path.join(model_root, dico_models[args.exp][data_path[args.number].split(".")[0]])

    main(el, train_folds, val_folds, test_folds, model_path)

