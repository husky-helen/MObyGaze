import argparse
import ast

def str_to_list(s):
    return ast.literal_eval(s)

def str_to_bool(s):
    """
    Fonction pour convertir une chaîne en booléen.
    """
    s = s.lower()
    if s in ('true', 't', '1', 'oui', 'vrai'):
        return True
    elif s in ('false', 'f', '0', 'non', 'faux'):
        return False
    else:
        raise argparse.ArgumentTypeError('Valeur booléenne invalide : {}'.format(s))
    

parser = argparse.ArgumentParser(description='WSAD')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=16, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
parser.add_argument('--annotation_path', type=str, default="/home/julie/Code/WSL/Preprocess/EN_S_main_lucile.csv", help='where to store the results and models')
parser.add_argument('--saving_dir', type=str, default="/media/LaCie/ResultsDeepMIL/ObyGaze", help='where to store the results and models')
parser.add_argument('--feature_dir', type=str, default="/media/LaCie/Features/Clip_16_Frames", help='feature directory')

parser.add_argument('--nb_segment', type=int, default=16, help='number of segment in one video')
parser.add_argument('--feature_dim', type=int, default=512, help='dimension of the feature that have been extracted from the extractor (XCLIP)')
parser.add_argument('--test_only', type=str_to_bool, default=False, help='test or train')
parser.add_argument('--oversample', type=str_to_bool, default=False, help='test or train')
parser.add_argument('--anomaly_classes', type=str_to_list, default=["Sure"], help='where to store the results and models')
parser.add_argument('--patience', type=int, default=5, help='patience allowed in early stopping')
parser.add_argument('--early_stopping', type=str_to_bool, default=True, help='Wether we should apply early stopping or not')
parser.add_argument('--do', type=float, default=0.2, help='drop out')
parser.add_argument('--scheduler', type=str, default='ReduceLR', help='set the scheduler to use to reduce the learning weight')

parser.add_argument('--val_fold_number', type=int, default=8, help='Number of the fold to use as validation set')
parser.add_argument('--test_fold_number', type=int, default=9, help='Number of the fold to use as test set')
parser.add_argument('--infer_annotation_path', type=str, default=None, help='annotation file to use to infer')
parser.add_argument('--model-name', default='deepmil_test1_', help='name to save model')
parser.add_argument('--max-epoch', type=int, default=50, help='maximum iteration to train (default: 100)')