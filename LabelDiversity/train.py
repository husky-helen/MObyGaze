# python imports
import argparse
import os
import time
import datetime
from pprint import pprint
import ast
import json

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# our code
from libs.core import load_config
from libs.dataset import make_dataset, make_data_loader
from libs.dataset import tractive
from libs.modeling import make_backbone, soft_loss, inverted_variety_loss
from libs.utils import (train_one_epoch, valid_one_epoch,valid_one_epoch_bms, 
                        save_checkpoint,
                        fix_random_seed)

################################################################################
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def main(args):
    
    """main function that handles training / inference"""

            
    #SET THE GPU 
    torch.cuda.set_device(0)  #based on the cuda environment visibility
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Training with GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

    """------------------ 1. setup parameters / folders ------------------"""

    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    
    
    exp_config = cfg['exp_config']
    yaml_name =  cfg['yaml_name']
    
    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
        print("Makedir output folder line 42 : ",cfg['output_folder'] )
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
      
        ckpt_folder = os.path.join(cfg['output_folder'], str(args.output))
    
    val_folds = args.valfold
    test_folds = args.testfold
    
    #new folder tree needs these parameter ckpt/tractive/{experiment_type}/{yaml_name}/{val_fold}/{test_fold}/
    if exp_config and yaml_name:
        if val_folds and test_folds: 
            #val_folds_dir_name = s.strip("[]'").replace("'", "")
            val_ids = ast.literal_eval(val_folds)
            val_dir = '_'.join(val_ids)
            
            test_ids = ast.literal_eval(test_folds)
            test_dir = '_'.join(test_ids)
            ckpt_folder = os.path.join(cfg['output_folder'], str(args.output), exp_config, yaml_name, "val_"+ val_dir, "test_"+ test_dir)
            
        else: 
            ckpt_folder = os.path.join(cfg['output_folder'], str(args.output), exp_config, yaml_name)
            
        print("CKPT FULL FOLDER")
        print(ckpt_folder)
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder, exist_ok=True)
    else:
        
        if not os.path.exists(ckpt_folder): os.mkdir(ckpt_folder)


    ckpt_folder =  os.path.join(cfg['output_folder'], exp_config, yaml_name, args.fnb)
    if not os.path.exists(ckpt_folder): os.makedirs(ckpt_folder)
    print("ckt folder : ", ckpt_folder)
  
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """------------------ 2. create dataset / dataloader ------------------"""

    val_folds = ast.literal_eval(args.valfold)
    test_folds = ast.literal_eval(args.testfold)


    print("VAL FOLDS : ", val_folds)
    print("TEST FOLDS : ", test_folds)

    train_dataset = make_dataset(cfg['dataset_name'], cfg['train_split'], **cfg['dataset'], val_folds=val_folds, test_folds=test_folds)   
    val_dataset = make_dataset(cfg['dataset_name'], cfg['val_split'], **cfg['dataset'], val_folds=val_folds, test_folds=test_folds)

    print("LEN train dataset : ", len(train_dataset))
    print("LEN val dataset : ", len(val_dataset))
  
    # data loaders
    train_loader = make_data_loader(train_dataset, True, rng_generator, **cfg['loader'])
    val_loader = make_data_loader(val_dataset, False, None, **cfg['loader'])


    output_file = os.path.join(ckpt_folder, 'eval_results.pkl')



    """------------------ 3. create model, optimizer, and scheduler ------------------"""
    # model
    if "soft" in cfg["loss"]:
        model = make_backbone(cfg['model_name'], **cfg['model'], loss_function=soft_loss)
    elif "inverted" in cfg["loss"]:
        model = make_backbone(cfg['model_name'], **cfg['model'], loss_function=inverted_variety_loss)
    else:
        model = make_backbone(cfg['model_name'], **cfg['model'])


    model = nn.DataParallel(model, device_ids=cfg['devices'])
    optimizer = Adam(model.parameters(), lr = cfg['opt']["learning_rate"], weight_decay =  cfg['opt']["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', patience=cfg['opt']['patience'])


    """------------------ 4. Resume from model / Misc ------------------"""

    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """------------------ 5. training / validation loop ------------------"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] 
    )

    # EARLY STOPPING PARAMETERS
    best_val_loss = 10000
    early_stopping_limit = cfg["train_cfg"]["early_stopping"]
    early_stopping_counter = 0
    val_losses = []

    all_evaluation_results = {}
    

    ckpt_status_folder = os.path.join(ckpt_folder, "ckpt")
    if not os.path.exists(ckpt_status_folder): os.mkdir(ckpt_status_folder)
    
    for epoch in range(args.start_epoch, max_epochs):

        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )

        # save ckpt once in a while
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_checkpoint(
                save_states,
                False,
                file_folder= ckpt_status_folder, #ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )
        if 'bms' in cfg["dataset_name"]: 
            losses, results_dico = valid_one_epoch_bms(
                val_loader,
                model,
                epoch,
                output_file=output_file,
                tb_writer = tb_writer,
                print_freq = args.print_freq,
                dir_writer='validation')

        else:
            losses, results_dico = valid_one_epoch(
                val_loader,
                model,
                epoch,
                output_file=output_file,
                tb_writer = tb_writer,
                print_freq = args.print_freq,
                dir_writer='validation')
        
        val_final_loss = losses['final_loss']
        val_losses.append(val_final_loss)
        
        all_evaluation_results[f'epoch_{epoch}'] = {
        'metrics': results_dico
        }   


        
        json_storage_path = os.path.join(ckpt_folder, "results")
        
        if not os.path.exists(json_storage_path): os.mkdir(json_storage_path)
        
        results_file_path = os.path.join(json_storage_path, f"validation_results_{yaml_name}_{args.output}.json")
        
        with open(results_file_path, 'w') as file:
            json.dump(all_evaluation_results, file, indent=4)
           
        # EARLY STOPPING
        if val_final_loss<=best_val_loss: 
            best_val_loss = val_final_loss
            early_stopping_counter = 0
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_status_folder, #ckpt_folder,
                file_name='best_model.pth.tar'.format(epoch + 1)
            )


        else: early_stopping_counter +=1

        if early_stopping_counter >= early_stopping_limit:
            print("TRAINING STOPPED AT EPOCH ", epoch, " BECAUSE OF EARLY STOPPING")
            break

    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)') 
    parser.add_argument('-valfold', type=str, default="['tt0822832']")
    parser.add_argument('-testfold', type=str, default="['tt0822832']")
    parser.add_argument('-fnb', '--fnb', type=str, default="fold_1")
    args = parser.parse_args()
    main(args)
