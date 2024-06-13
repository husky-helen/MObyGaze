import torch
import os
import ast
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from sklearn.model_selection import train_test_split


class LinearProbing_KFold(Dataset):
    def __init__(self, root_dir, gt_csv, split="train", test_fold = None, val_fold = None, dico_level =  {'Easy Negative':0,'Hard Negative':1, "Not Sure":2, "Sure":3}, movie_ids = None, auto_split = False):
       
        assert split in ["train", "val", "test"]
        
        self.root_dir = root_dir  #root directory where the movie_id folders are /root/{movie_id} #/media/LaCie/Features/clips_features
        self.gt_csv = gt_csv
        self.split = split
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.dico_level = dico_level
        self.movie_ids = movie_ids
        self.auto_split = auto_split

        self.gt_df = pd.read_csv(self.gt_csv, index_col=0,sep=";")

        self.get_label()
        self.weight = self.compute_weight()
        self.getsplit()

    def get_label(self):
        new_label = []
        for index, row in self.gt_df.iterrows():
            new_label.append(self.dico_level[row.label])
        self.gt_df["label"] = new_label


    def compute_weight(self):
        max_fold = self.gt_df.fold.max()
        sub_df = self.gt_df.query("fold!=@max_fold").label
        return torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(sub_df), y=sub_df))
    
    def balance_classes(self, df): #balance minority class to match majority one with random oversampling
        """Dynamically balance the classes by oversampling the minority class."""
        label_counts = df.label.value_counts()
        max_count = label_counts.max()

        #oversample each class to have the same number of samples as the max_count
        balanced_df = pd.concat([df[df.label == label].sample(n=max_count, replace=True) for label in label_counts.index])
        
        #shuffle the dataframe to mix the oversampled entries
        balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
        

        return balanced_df
        
        
    def getsplit(self):
        
        if self.split == "test":
            self.gt_df = self.gt_df.query("fold == @self.test_fold")
        elif self.split == "val":
            self.gt_df = self.gt_df.query("fold == @self.val_fold")
        else:
            # For training, exclude both test and validation folds
            self.gt_df = self.gt_df.query("fold != @self.test_fold and fold != @self.val_fold")

           
            label_counts = self.gt_df.label.value_counts()
            
            for label, count in label_counts.items():
                label_name = list(self.dico_level.keys())[list(self.dico_level.values()).index(label)]  
                

            self.gt_df = self.balance_classes(self.gt_df) #balanced dataframe
            
            label_counts_after = self.gt_df.label.value_counts()
            
            for label, count in label_counts_after.items():
                label_name = list(self.dico_level.keys())[list(self.dico_level.values()).index(label)]  
                
            
    def __len__(self):
        return self.gt_df.shape[0]
    
    def __getitem__(self, idx):
        #setting max attemps for indexing error
        max_attempts = 10 
        file_found = False
        
      
        for attempts in range(max_attempts):
            if file_found:
                break 
        
            
            gt_data = self.gt_df.iloc[idx,:] #takes the annotations at the extracted row (idx) to process it
            
            feature_index = gt_data.clip_index  #is the clip index, which have to be mapped tt..._1
            movie_id = gt_data.imdb_key          #imdb key
            label = gt_data.label
            annotator = gt_data.annotator
            one_hot_encoded = F.one_hot(torch.tensor(label), len(self.dico_level)).float()

            #visual_dir = root directory /root/{movie_id} -> /media/LaCie/Features/clip_features/...
            movie_dir = os.path.join(self.root_dir, movie_id)

      
            video_features = None
            
            for annotator_name in os.listdir(movie_dir): #iterate through the different annotators inside the movie directory
                annotator_dir = os.path.join(movie_dir, annotator_name)
                
                if os.path.isdir(annotator_dir): 
                    
                    if annotator == annotator_name:
                        feature_file_path = os.path.join(annotator_dir, f"{feature_index}.pt") #old code was pth

                        if os.path.exists(feature_file_path): 
                            
                            video_features = torch.load(feature_file_path, map_location="cpu")
                            
                            
                            video_features.requires_grad = False
                            shape_vf = video_features.shape
                            if len(shape_vf) > 2:
                                dim = shape_vf[-1]
                                m = nn.AdaptiveMaxPool2d((1, dim))
                                video_features = m(video_features)[0][0]
                            else:
                                video_features = video_features[0]
                            file_found = True
                            break
                
            if not file_found:
                print("FILE NOT FOUND")
                idx = torch.randint(low=0, high=self.__len__(), size=(1,)).item()

        
        #Raise recursion error as it doesn't find any feature
        if not file_found:
            raise FileNotFoundError(f"Unable to find a valid pt feature after {max_attempts} attempts.")

       
        return video_features, one_hot_encoded
