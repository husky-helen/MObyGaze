import torch.utils.data as data
import numpy as np
from .utils import  load_tensor_list
import torch
import torch.nn as nn

import os
import pandas as pd
from .dataset import register_dataset


@register_dataset("tractive")
class TractiveDataset(data.Dataset):

    def __init__(self, 
                 mode , 
                 feature_dir = "/media/LaCie/Features",
                 feature_stride = 16,
                 feature_window = 16,
                 nb_segment = 16,
                 annotation_path = 'annotation.csv', 
                 dico_label = {"Easy Negative":0, "Hard Negative":1, "Not Sure":1, "Sure":1},
                 val_folds = ["tt0822832"], 
                 test_folds = ["tt0822832"], 
                 oversampling = False):
        
        assert mode in ["train", "validation", "test"]
        assert os.path.exists(feature_dir)

        self.feature_dir = feature_dir
        self.feature_stride = feature_stride
        self.feature_window = feature_window

        self.oversampling = oversampling

        self.nb_segment = nb_segment
        self.mode = mode
        self.dico_label = dico_label
        self.val_folds = val_folds
        self.test_folds = test_folds

        self.annotation_path = annotation_path
        self.load_df()
        self.pooling = nn.AdaptiveAvgPool1d(1)

    
    def load_df(self):

        if not os.path.exists(self.annotation_path): assert False
        annotation_df = pd.read_csv(self.annotation_path, sep=";", index_col=0)
        
        assert 'imdb_key' in annotation_df.columns
        if self.mode == 'train':
            self.annotation_df = annotation_df.query("imdb_key not in @self.val_folds and imdb_key not in @self.test_folds")

            if self.oversampling:
                self.annotation_df["intlabel"] = self.annotation_df["label"].map(self.dico_label)

                df_neg = self.annotation_df.query("intlabel == 0")
                df_pos = self.annotation_df.query("intlabel == 1")
                nb_neg = df_neg.shape[0]
                nb_pos = df_pos.shape[0]


                if nb_neg > nb_pos: # Not enough positive data
                    to_add = nb_neg - nb_pos
                    idx = np.random.randint(0,nb_pos, to_add)
                    df_toadd = df_pos.iloc[idx,:]
                    
                elif nb_pos > nb_neg:
                    to_add = nb_pos - nb_neg
                    idx = np.random.randint(0,nb_neg, to_add)
                    df_toadd = df_neg.iloc[idx,:]
                   
                self.annotation_df = pd.concat([self.annotation_df, df_toadd], axis = 0)
                df_neg = self.annotation_df.query("intlabel == 0")
                df_pos = self.annotation_df.query("intlabel == 1")
                nb_neg = df_neg.shape[0]
                nb_pos = df_pos.shape[0]


        elif self.mode == "validation":
            self.annotation_df = annotation_df.query("imdb_key in @self.val_folds")
        elif self.mode == "test":
            self.annotation_df = annotation_df.query("imdb_key in @self.test_folds")
        self.annotation_df.reset_index(inplace=True, drop=True)

    def __len__(self):
        return self.annotation_df.shape[0]
    
    def __getitem__(self, index):

        row = self.annotation_df.iloc[index,:]
        movie_imdb = row.imdb_key 
        
        starting_feat = row.start_frame // self.feature_window
        ending_feat = row.end_frame // self.feature_stride
        
        
        features_path = [os.path.join(self.feature_dir, movie_imdb, movie_imdb + "_"+ str(clip_number) + ".pt")  for clip_number in range(starting_feat, ending_feat + 1)if os.path.exists(os.path.join(self.feature_dir, movie_imdb, movie_imdb + "_"+ str(clip_number) + ".pt"))] 
        features = load_tensor_list(features_path).detach()
        

        if len(features.shape)>1: features = torch.max(features, dim=0).values
        label = self.dico_label[row.label]
    
        return features, label
    

 