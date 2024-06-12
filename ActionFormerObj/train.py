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

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.datasets import tractive
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)

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
        
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
   
        ckpt_folder = os.path.join(cfg['output_folder'], str(args.output))
    
    val_folds = args.valfold
    test_folds = args.testfold
    


    ckpt_folder =  os.path.join(cfg['output_folder'], exp_config, yaml_name, args.fnb)
    if not os.path.exists(ckpt_folder): os.makedirs(ckpt_folder)


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

    train_dataset = make_dataset(cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset'], val_folds=val_folds, test_folds=test_folds)
    val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'], val_folds=val_folds, test_folds=test_folds)
  
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

 
    print("LEN train dataset : ", len(train_dataset))
    print("LEN val dataset : ", len(val_dataset))


    train_loader = make_data_loader(train_dataset, True, rng_generator, **cfg['loader'])
    val_loader = make_data_loader(val_dataset, False, None, **cfg['val_loader'])
    
    det_eval, output_file = None, None

    if not args.saveonly:
        
        val_db_vars = val_dataset.get_attributes()

        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds'], 
            val_folds=val_folds, 
            test_folds=test_folds
        )
    else:
        output_file = os.path.join(ckpt_folder, 'eval_results.pkl')



    """------------------ 3. create model, optimizer, and scheduler ------------------"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    
 
    optimizer = make_optimizer(model, cfg['opt'])
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """------------------ 4. Resume from model / Misc ------------------"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
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
    max_epochs = cfg['opt'].get('early_stop_epochs',cfg['opt']['epochs'] + cfg['opt']['warmup_epochs'])

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
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
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

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder= ckpt_status_folder, #ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )

        losses, mAP, results_logs = valid_one_epoch(
            val_loader,
            model,
            epoch,
            evaluator=det_eval,
            output_file=output_file,
            ext_score_file=cfg['test_cfg']['ext_score_file'],
            tb_writer = tb_writer,
            print_freq = args.print_freq)
        val_final_loss = losses['final_loss']
        val_losses.append(val_final_loss)
        
        all_evaluation_results[f'epoch_{epoch}'] = {
        'metrics': results_logs
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

            save_states['state_dict_ema'] = model_ema.module.state_dict()
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
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)') 
    parser.add_argument('-valfold', type=str, default=None)
    parser.add_argument('-testfold', type=str, default=None)
    parser.add_argument('-fnb', '--fnb', type=str, default="fold_1")
    args = parser.parse_args()
    main(args)
