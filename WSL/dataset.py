import torch.utils.data as data
import numpy as np
from utils import process_feat, load_tensor_list
import torch
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
import os
from collections import Counter
import pandas as pd


   


class DatasetFolds(data.Dataset):
    def __init__(self, args, is_normal = True,mode="train", oversampling=False, anomaly_classes = ["Sure"], val_fold_number = 8, test_fold_number=9):
        """

        Args:
            args : global argumetns
            mode (str, optional): loads the dataset according to the mode. Defaults to "train".
            oversampling (bool, optional): If true, the minor class between anomaly and normal is oversampled. Default to False
        """

        assert mode in ["train", "val", "test"]
        self.is_normal = is_normal
        self.mode = mode
        self.oversampling = oversampling
        self.anomaly_classes = anomaly_classes

        self.annotation_path = args.annotation_path
        
        self.test_fold_number = test_fold_number
        self.val_fold_number = val_fold_number



        self.get_annotation()
    
        
        self.feature_dir = args.feature_dir
        self.nb_segment = args.nb_segment
        self.feature_dim = args.feature_dim

    def get_annotation(self):

        self.annotation_df = pd.read_csv(self.annotation_path, sep=";", index_col=0)
        np.random.seed(42)

        if self.mode == "test":
            self.annotation_df = self.annotation_df.query("fold == @self.test_fold_number")

        elif self.mode == 'val':
            self.annotation_df = self.annotation_df.query("fold == @self.val_fold_number")
        
        else:
            self.annotation_df = self.annotation_df.query("fold != @self.val_fold_number and fold != @self.test_fold_number")
            normal = self.annotation_df.query("not label in @self.anomaly_classes")
            anomaly = self.annotation_df.query("label in @self.anomaly_classes")

            if self.oversampling:
                nb_normal = normal.shape[0]
                nb_anomaly = anomaly.shape[0]

                if nb_normal > nb_anomaly:
                    to_add = nb_normal - nb_anomaly
                    idx = np.random.randint(0,nb_anomaly, to_add)
                    tmp_anomaly = anomaly.iloc[idx,:]
                    anomaly = pd.concat([anomaly, tmp_anomaly], axis=0)
                elif nb_anomaly > nb_normal:
                    to_add = nb_anomaly - nb_normal
                    idx = np.random.randint(0,nb_normal, to_add)
                    tmp_normal = normal.iloc[idx,:]
                    normal = pd.concat([normal, tmp_normal], axis=0)

            if self.is_normal:
                self.annotation_df = normal
            else:
                self.annotation_df = anomaly

    def __getitem__(self, index):

        row = self.annotation_df.iloc[index,:]
        movie_imdb = row.imdb_key 
        
        starting_clip = row.start_clip
        ending_clip = row.end_clip

        features_path = [os.path.join(self.feature_dir, movie_imdb, movie_imdb + "_"+ str(clip_number) + ".pt")  for clip_number in range(starting_clip, ending_clip + 1)if os.path.exists(os.path.join(self.feature_dir, movie_imdb, movie_imdb + "_"+ str(clip_number) + ".pt"))] 
        features = load_tensor_list(features_path,self.feature_dim).detach().numpy()
        
        features = process_feat(features, self.nb_segment)
        if self.mode != "train":
            if row.label in self.anomaly_classes: label = 1
            else: label = 0
            return features, label
        return features


    def __len__(self):
        if self.mode == "train":
            if self.oversampling:
                normal= self.annotation_df.query("not label in @self.anomaly_classes")
                anomaly = self.annotation_df.query("label in @self.anomaly_classes")
                return max(normal.shape[0], anomaly.shape[0])
        
        return self.annotation_df.shape[0]
    

    def __init__(self, args, is_normal = True,mode="train", oversampling=False, anomaly_classes = ["Sure"], val_fold_number = 8, test_fold_number=9):
        """TODO : le paramètre "annotation dir" doit être dans les args

        Args:
            args (_type_): global argumetns
            mode (str, optional): loads the dataset according to the mode. Defaults to "train".
            oversampling (bool, optional): If true, the minor class between anomaly and normal is oversampled. Default to False
        """
        assert mode in ["train", "val", "test"]
        self.is_normal = is_normal
        self.mode = mode
        self.oversampling = oversampling
        self.anomaly_classes = anomaly_classes

        if args.infer_annotation_path != None: self.annotation_path = args.infer_annotation_path
        else: self.annotation_path = args.annotation_path
        
        self.test_fold_number = test_fold_number
        self.val_fold_number = val_fold_number



        self.get_annotation()
    
        
        self.feature_dir = args.feature_dir
        self.nb_segment = args.nb_segment
        self.feature_dim = args.feature_dim

    def get_annotation(self):
        self.annotation_df = pd.read_csv(self.annotation_path, sep=";", index_col=0)
        np.random.seed(42)

        if self.mode == "test":
            self.annotation_df = self.annotation_df.query("fold == @self.test_fold_number")

        elif self.mode == 'val':
            self.annotation_df = self.annotation_df.query("fold == @self.val_fold_number")
        
        else:
            self.annotation_df = self.annotation_df.query("fold != @self.val_fold_number and fold != @self.test_fold_number")
            print("Repartition : ", Counter(self.annotation_df.label))
            normal = self.annotation_df.query("not label in @self.anomaly_classes")
            anomaly = self.annotation_df.query("label in @self.anomaly_classes")

            if self.oversampling:
                nb_normal = normal.shape[0]
                nb_anomaly = anomaly.shape[0]

                if nb_normal > nb_anomaly:
                    to_add = nb_normal - nb_anomaly
                    idx = np.random.randint(0,nb_anomaly, to_add)
                    tmp_anomaly = anomaly.iloc[idx,:]
                    anomaly = pd.concat([anomaly, tmp_anomaly], axis=0)
                elif nb_anomaly > nb_normal:
                    to_add = nb_anomaly - nb_normal
                    idx = np.random.randint(0,nb_normal, to_add)
                    tmp_normal = normal.iloc[idx,:]
                    normal = pd.concat([normal, tmp_normal], axis=0)

            if self.is_normal:
                self.annotation_df = normal
            else:
                self.annotation_df = anomaly

    def __getitem__(self, index):

        row = self.annotation_df.iloc[index,:]
        movie_imdb = row.imdb_key 
        video_index = row.clip_index 
        
        starting_clip = row.start_clip
        ending_clip = row.end_clip
        #print("S-E : ", starting_clip, ending_clip)
        features_path = [os.path.join(self.feature_dir, movie_imdb, movie_imdb + "_"+ str(clip_number) + ".pt")  for clip_number in range(starting_clip, ending_clip + 1)if os.path.exists(os.path.join(self.feature_dir, movie_imdb, movie_imdb + "_"+ str(clip_number) + ".pt"))] 
        features = load_tensor_list(features_path,self.feature_dim).detach().numpy()
        
        #print(features_path)
        features = process_feat(features, self.nb_segment)
        if self.mode != "train":
            if row.label in self.anomaly_classes: label = 1
            else: label = 0
            return features, label
        return features


    def __len__(self):
        if self.mode == "train":
            if self.oversampling:
                normal= self.annotation_df.query("not label in @self.anomaly_classes")
                anomaly = self.annotation_df.query("label in @self.anomaly_classes")
                return max(normal.shape[0], anomaly.shape[0])
        
        return self.annotation_df.shape[0]