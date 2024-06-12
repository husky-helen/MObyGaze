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
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import test_one_epoch, ANETdetection, fix_random_seed


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

    """if torch.cuda.is_available():
        print("Num GPUs Available:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")"""
            
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
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    

    """ Old code
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
        
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:

            if os.path.exists(os.path.join(args.ckpt, 'best_model.pth.tar')): ckpt_file = os.path.join(args.ckpt, 'best_model.pth.tar')
            else:
                ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
                print("ARGS CKPT : ", args.ckpt)
                ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file) """
    
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "Directory does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, f'epoch_{args.epoch:03d}.pth.tar'
            )
        else:
            #try to find the 'ckpt' directory if it's not directly provided
            ckpt_dir = find_ckpt_dir(args.ckpt)
            if ckpt_dir:
                args.ckpt = ckpt_dir
            if os.path.exists(os.path.join(args.ckpt, 'best_model.pth.tar')):
                ckpt_file = os.path.join(args.ckpt, 'best_model.pth.tar')
            else:
                ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
                if ckpt_file_list:
                    ckpt_file = ckpt_file_list[-1]
                else:
                    raise FileNotFoundError("No checkpoint files found in the directory.")
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """------------------ 1. fix all randomness ------------------"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """------------------ 2. create dataset / dataloader ------------------"""

    json_fold_path = "/media/LaCie/Annotation_20240521/Folds/folds.json"
    with open(json_fold_path, 'r') as f:
        dico_folds = json.load(f)
    val_folds = dico_folds[args.fnb]["validation"]
    test_folds = dico_folds[args.fnb]["test"]



    """val_folds = args.valfold
    test_folds = args.testfold
    if val_folds != None: val_folds = ast.literal_eval(val_folds)
    if test_folds != None: test_folds = ast.literal_eval(test_folds)
    print("VAL FOLDS : ", val_folds)
    print("TEST FOLDS : ", test_folds)"""

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['test_split'], **cfg['dataset'], val_folds=val_folds, test_folds=test_folds
    )
    print("LEN test dataset : ", len(val_dataset))
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )
    



    """------------------ 3. create model and evaluator ------------------"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # set up evaluator
    det_eval, output_file = None, None
    print("ANETdetection")
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
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """------------------ 5. Test the model ------------------"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    
    start = time.time()
    mAP, results_logs = test_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    
    all_test_results = {
    'metrics': results_logs
    }   
        
    parent_dir = os.path.dirname(args.ckpt)
    results_dir = os.path.join(parent_dir, 'results')
    
    print(results_dir)
    #get filename
    json_file_name = cfg['yaml_name']
    
    results_file_path = os.path.join(results_dir, f"test_results_{json_file_name}.json")
    
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
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-valfold', type=str, default=None)
    parser.add_argument('-testfold', type=str, default=None)
    parser.add_argument('-fnb', '--fnb', type=str, default="fold_1")
    #parser.add_argument('--output', default='', type=str, help='name of exp folder (default: none)')
    args = parser.parse_args()
    main(args)
