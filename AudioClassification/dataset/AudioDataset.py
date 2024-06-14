import os
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoProcessor
import torchaudio.transforms
import numpy as np
import torch
import os
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms
import numpy as np

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torchaudio.transforms as T

import torch

class AudioDataset(Dataset):
    def __init__(self, csv_file, feature_dir, mode='train', train_folds=None, val_fold=None, test_fold=None):
        
        """
        Initialize the dataset.
        Args:
        - csv_file: Path to the CSV file containing the dataset metadata.
        - base_dir: Base directory where the audio files are stored.
        - mode: One of 'train', 'val', or 'test' to set the operating mode.
        """
        
        self.data = pd.read_csv(csv_file, delimiter=';')

        self.feature_dir = feature_dir
        
        self.train_folds = train_folds
        self.val_fold = val_fold
        self.test_fold = test_fold
        
        self.mode = mode

        self.getsplit()
  
    def getsplit(self):
        
        if self.mode == 'train' and self.train_folds is not None:
            self.data = self.data[self.data['fold'].isin(self.train_folds)]
            print(f"Original training data distribution (folds {self.train_folds}) :\n{self.data['label_audio'].value_counts()}\n")
            #balancing
            self.data = self.balance_classes(self.data)
            print(f"New balanced training data distribution (folds {self.train_folds})  :\n{self.data['label_audio'].value_counts()}\n")
            
        elif self.mode == 'val' and self.val_fold is not None:
            self.data = self.data[self.data['fold'] == self.val_fold]
        elif self.mode == 'test' and self.test_fold is not None:
            self.data = self.data[self.data['fold'] == self.test_fold]
    
    def balance_classes(self, df): 
        """Dynamically balance the classes by oversampling the minority class."""
        label_counts = df['label_audio'].value_counts()
        max_count = label_counts.max()
        #oversample each class to have the same number of samples as the max_count
        balanced_df = pd.concat([df[df['label_audio'] == label].sample(n=max_count, replace=True) for label in label_counts.index])

        #shuffle the dataframe to mix the oversampled entries
        balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
        
        return balanced_df

    def __len__(self):
        return len(self.data)
          
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        #speech_index is a temporarily created index that has the same role of clip_index (since in the data pre processing we split the data into 60seconds chuncks the clip index was repeated, so speech_index act as the new identifier)
        speech_index = row['speech_index'].split('_')[1]
        
        feature_path = os.path.join(self.feature_dir, row['imdb_key'], f"{row['imdb_key']}_{speech_index}_{row['annotator']}.pt")
        
        if os.path.exists(feature_path):
            inputs = torch.load(feature_path)
        else:
            print(f"feature path {feature_path} not found. Trying with another index...")
            found = False
            for alternate_idx in range(len(self.data)):
                if alternate_idx == idx:
                    continue  #skip the current index
                
                alternate_row = self.data.iloc[alternate_idx]
                alternate_speech_index = alternate_row['speech_index'].split('_')[1]
                alternate_feature_path = os.path.join(self.feature_dir, alternate_row['imdb_key'], f"{alternate_row['imdb_key']}_{alternate_speech_index}_{alternate_row['annotator']}.pt")
                
                if os.path.exists(alternate_feature_path):
                    #print(f"Found alternate feature path {alternate_feature_path}")
                    inputs = torch.load(alternate_feature_path)
                    found = True
                    break

            if not found:
                raise FileNotFoundError(f"No valid feature path found for index {idx}")
            
        
    
        label = torch.tensor(row['label_audio'], dtype=torch.float16)
        
        return inputs, label
