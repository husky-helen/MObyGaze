# python imports
import argparse
import os
import glob
import time
from pprint import pprint
import ast
import json

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data


# our code
from libs.core import load_config
from libs.dataset import make_dataset, make_data_loader
from libs.modeling import make_backbone, soft_loss, inverted_variety_loss
from libs.utils import valid_one_epoch, fix_random_seed
from torch.utils.tensorboard import SummaryWriter

################################################################################

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
 
def find_ckpt_dir(root_dir): #walk the directories till it finds the ckpt folder that contains .tar files 
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'ckpt' in dirnames:
            ckpt_path = os.path.join(dirpath, 'ckpt')
            if any(f.endswith('.tar') for f in os.listdir(ckpt_path)):
                return ckpt_path
    return None

def main(args):

   
            
    #SET THE GPU 
    torch.cuda.set_device(0) #based on the cuda environment visibility
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Evaluating with GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    """------------------ 0. load config ------------------"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")



    """------------------ 1. fix all randomness ------------------"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)
    

    """------------------ 2. create dataset / dataloader ------------------"""
    val_folds = ast.literal_eval(args.valfold)
    test_folds = ast.literal_eval(args.testfold)
  
    print("VAL FOLDS : ", val_folds)
    print("TEST FOLDS : ", test_folds)

    val_dataset = make_dataset(cfg['dataset_name'], cfg['test_split'], **cfg['dataset'], val_folds=val_folds, test_folds=test_folds)
    print("LEN test dataset : ", len(val_dataset))
 

    val_loader = make_data_loader(val_dataset, False, None, 1, cfg['loader']['num_workers'])
    
    ckpt_folder = args.ckpt
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))
    ckpt_file = os.path.join(ckpt_folder,'ckpt','best_model.pth.tar' )
    
    
    """------------------ 3. create model and evaluator ------------------"""
    # model
    if "soft" in cfg["loss"]:
        model = make_backbone(cfg['model_name'], **cfg['model'], loss_function=soft_loss)
    elif "inverted" in cfg["loss"]:
        model = make_backbone(cfg['model_name'], **cfg['model'], loss_function=inverted_variety_loss)
    else:
        model = make_backbone(cfg['model_name'], **cfg['model'])


    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """------------------ 4. load ckpt ------------------"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )

    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint

    output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """------------------ 5. Test the model ------------------"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    
    start = time.time()
    losses, results_dico = valid_one_epoch(
        val_loader,
        model,
        -1,
        output_file=output_file,
        tb_writer=tb_writer,
        print_freq=args.print_freq,
        dir_writer='test'
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    
    all_test_results = {
    'metrics': results_dico
    }   
        
    results_dir = os.path.join(args.ckpt, "val_"+ val_folds[0], "test_"+ test_folds[0], 'results')
    json_file_name = cfg['yaml_name']
    results_file_path = os.path.join(results_dir, args.test_prefix + f"test_results_{json_file_name}.json")
    
    with open(results_file_path, 'w') as file:
        json.dump(all_test_results, file, indent=4)


    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-valfold', type=str, default=None)
    parser.add_argument('-testfold', type=str, default=None)
    parser.add_argument('-test_prefix', type=str, default="")
    parser.add_argument('-fnb', '--fnb', type=str, default="fold_1")
    #parser.add_argument('--output', default='', type=str, help='name of exp folder (default: none)')
    args = parser.parse_args()
    main(args)
