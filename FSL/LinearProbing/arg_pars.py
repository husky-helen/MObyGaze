import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--features_dir', default="/home/xxx/ssd/data1/Features/XCLIP/Visual")
parser.add_argument('--annotation_dir', default= "/home/xxx/Code/CleanCVPR/LinearProbing/models_movieset/annotation_files")
parser.add_argument('--training_saving_dir', default="/home/xxx/Results/TrainingMoviesSeparated/")
parser.add_argument('--model_type', default="XCLIP", help="XCLIP or ViVit or XCLIP_LSMDC")
parser.add_argument('--checkpoints', default =  "/home/xxx/Results/TrainingMoviesSeparated/")
parser.add_argument('--infering_saving_dir', default="/home/xxx/Results/InferenceMoviesSeparated/")
opt = parser.parse_args()
