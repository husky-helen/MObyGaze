import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torch.utils.data import Dataset

class SpeechDataset(Dataset):
    def __init__(self, csv_file, train_folds=None, val_fold=None, test_fold=None, mode='train'):
        """
        Initializes the dataset with specific folds for training, validation, and testing.
        Args:
        - csv_file: Path to the CSV file containing the dataset.
        - train_folds: List of fold numbers to include in the training set.
        - val_fold: Single fold number to use for validation.
        - test_fold: Single fold number to use for testing.
        - mode: One of 'train', 'val', or 'test' to set the operating mode.
        """
        
        self.csv_file = csv_file
        self.train_folds = train_folds
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.mode = mode
        
        self.data = pd.read_csv(csv_file, sep=";")
        
        #split + random oversampling of the minority class for training
        self.getsplit()

       
    def getsplit(self):
        
        if self.mode == 'train' and self.train_folds is not None:
            self.data = self.data[self.data['fold'].isin(self.train_folds)]
            print(f"Original training data distribution (folds {self.train_folds}) :\n{self.data['label_speech'].value_counts()}\n")
            #balancing
            self.data = self.balance_classes(self.data)
            print(f"New balanced training data distribution (folds {self.train_folds})  :\n{self.data['label_speech'].value_counts()}\n")
            
        elif self.mode == 'val' and self.val_fold is not None:
            self.data = self.data[self.data['fold'] == self.val_fold]
        elif self.mode == 'test' and self.test_fold is not None:
            self.data = self.data[self.data['fold'] == self.test_fold]
            
            
    def balance_classes(self, df): 
        """Dynamically balance the classes by oversampling the minority class."""
        label_counts = df['label_speech'].value_counts()
        max_count = label_counts.max()
        #oversample each class to have the same number of samples as the max_count
        balanced_df = pd.concat([df[df['label_speech'] == label].sample(n=max_count, replace=True) for label in label_counts.index])

        #shuffle the dataframe to mix the oversampled entries
        balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
        
        return balanced_df
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        label = row['label_speech']
        
        return text, label