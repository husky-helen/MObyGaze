from torch.utils.data import DataLoader

import torch.optim as optim
import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import baselines, evaluate
import ast

def string_to_list(string):
    return ast.literal_eval(string)

def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  


setup_seed(int(42))  # 1577677170  2333

from model import ModelFinal
from WSL.dataset import  DatasetFolds
from train import train
from test import agg_test
import WSL.option as option



viz=None


if __name__ == '__main__':
    args = option.parser.parse_args()
    if isinstance(args.anomaly_classes, str):
        args.anomaly_classes = string_to_list(args.anomaly_classes)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    device = torch.device("cuda")  # TODO erase this (should be a parameter in args)
    print("Experiment Setting : ")
    print("Annotation Dir : ", args.annotation_path)
    print("Oversampling : ", args.oversample)
    print("Batch size : ", args.batch_size)
    print("Anomaly classes : ", args.anomaly_classes)
    print("Early stopping : ", args.early_stopping)
    print("Val fold number : ", args.val_fold_number)
    print("Test fold number : ", args.test_fold_number)

    if not args.early_stopping: patience = args.max_epoch
    else: patience = args.patience

    """ --------------   0. Data Initialization --------------- """
    train_ndataset = DatasetFolds(args, is_normal = True, mode="train", oversampling=args.oversample, anomaly_classes = args.anomaly_classes, val_fold_number = args.val_fold_number, test_fold_number=args.test_fold_number)
    train_adataset = DatasetFolds(args, is_normal = False, mode="train", oversampling=args.oversample, anomaly_classes = args.anomaly_classes, val_fold_number = args.val_fold_number, test_fold_number=args.test_fold_number)
    val_dataset = DatasetFolds(args, is_normal = True,mode="val", oversampling=args.oversample, anomaly_classes = args.anomaly_classes, val_fold_number = args.val_fold_number, test_fold_number=args.test_fold_number)
    test_dataset = DatasetFolds(args, is_normal = True,mode="test", oversampling=args.oversample, anomaly_classes = args.anomaly_classes, val_fold_number = args.val_fold_number, test_fold_number=args.test_fold_number)
    
  
    nb_ndataset = len(train_ndataset)
    nb_adataset = len(train_adataset)

    nb_iter_tmp = min(nb_ndataset, nb_adataset)

    
    train_nloader = DataLoader(train_ndataset,batch_size=args.batch_size, shuffle=True,num_workers=args.workers, pin_memory=True, drop_last = True)
    train_aloader = DataLoader(train_adataset,batch_size=args.batch_size, shuffle=True,num_workers=args.workers, pin_memory=True, drop_last = True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)


    if not args.test_only:
        """ --------------   1. Model Initialization --------------- """
        model = ModelFinal(args.feature_dim, dropout=args.do)
        model = model.to(device)

        best_val_metric = 0.
        early_stopping_counter = 0


        optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=0.00005)
        if args.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.1, verbose=True)
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        if not os.path.exists(os.path.join(args.saving_dir,"ckpt")):
            os.makedirs(os.path.join(args.saving_dir,"ckpt"))

        list_loss = []
        epoch=0

        """ --------------   2. Model Training--------------- """
        while epoch < args.max_epoch and early_stopping_counter!=patience:
            epoch += 1

            
            mean_loss = train(train_nloader, train_aloader, model, args.batch_size, optimizer, viz, device, nb_iter_tmp // args.batch_size, args.nb_segment)
            list_loss.append(mean_loss)
            
            if epoch % 1 == 0 and not epoch == 0:
                torch.save(model.state_dict(), os.path.join(args.saving_dir,"ckpt",args.model_name+'{}-xclip.pkl'.format(epoch)))
            
        
            dico_res = agg_test(val_loader, model, args, viz, device)
            print('Validation Epoch {0}/{1}: auc:{2:.2f}  |  f1:{3:.2f}  |  f1_w:{5:.2f}  |  accuracy:{4:.2f}'.format(epoch, args.max_epoch, dico_res["rec_auc"],dico_res["f1_score"], dico_res["accuracy"],dico_res["f1_score_w"]))
            if args.scheduler == "plateau":
                scheduler.step(dico_res["rec_auc"])
            else:
                scheduler.step()


            val_metric = dico_res["rec_auc"]

            if val_metric > best_val_metric:  
                best_val_metric = val_metric
                early_stopping_counter = 0
                
                torch.save(model.state_dict(), os.path.join(args.saving_dir,"ckpt",args.model_name + 'best_model.pth'))
            else:
                early_stopping_counter += 1

        torch.save(model.state_dict(), os.path.join(args.saving_dir,"ckpt",args.model_name + 'final.pkl'))


        """ --------------   3. Model Testing --------------- """
        dico_res = agg_test(test_loader, model, args, viz, device)
        print('Epoch {0}/{1}: auc:{2:.2f}  |  f1:{3:.2f}  |  f1_w:{5:.2f}  |  accuracy:{4:.2f}'.format(epoch, args.max_epoch, dico_res["rec_auc"],dico_res["f1_score"], dico_res["accuracy"],dico_res["f1_score_w"]))

        

    else:
        model = ModelFinal(args.feature_dim, dropout=args.do)

        checkpoint_path = os.path.join(args.saving_dir, 'ckpt',args.model_name + 'final.pkl' )
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        model = model.to(device)

        # val
        dico_res = agg_test(val_loader, model, args, viz, device)
        print('VAL Model : auc:{0:.2f}  |  f1:{1:.2f}  |  f1_w:{3:.2f}  |  accuracy:{2:.2f}  |  precision:{4:.2f}  |  recall:{5:.2f}'.format(dico_res["rec_auc"],dico_res["f1_score"], dico_res["accuracy"],dico_res["f1_score_w"], dico_res["precision_score"],dico_res["recall_score"]))

        # test 
        dico_res = agg_test(test_loader, model, args, viz, device)
        print('TEST Model : auc:{0:.2f}  |  f1:{1:.2f}  |  f1_w:{3:.2f}  |  accuracy:{2:.2f}  |  precision:{4:.2f}  |  recall:{5:.2f}'.format(dico_res["rec_auc"],dico_res["f1_score"], dico_res["accuracy"],dico_res["f1_score_w"], dico_res["precision_score"],dico_res["recall_score"]))

