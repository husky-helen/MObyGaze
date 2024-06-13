import torch
import os
import csv
import tqdm

def load_movie_features(feature_path): 
    feature_tensors = []
    for filename in os.listdir(feature_path):
        if filename.endswith('.pt'): 
            file_path = os.path.join(feature_path, filename)
            feature_tensor = torch.load(file_path)
            feature_tensors.append(feature_tensor)  
    return feature_tensors
        
IMDB_KEY = 'tt0097576'
feature_path = f'/media/LaCie/Features/clip_16_frames/{IMDB_KEY}/'  #path to a movie pt files 
saving_dir = "/media/LaCie/Features/Neurips_Features/clips_features_max"

annotations_path = "dataset/" 
annotations_csv = 'mobygaze_dataset.csv'
ann_path = os.path.join(annotations_path, annotations_csv)


feature_tensors = load_movie_features(feature_path)

aggregated_tensors = {}

with open(ann_path, mode='r', newline='') as file:

    reader = csv.reader(file, delimiter=';')
    headers = next(reader)

    for row in reader:
        if row[9] == IMDB_KEY:
            annotator = row[3]
          
            
            if annotator not in aggregated_tensors:
                aggregated_tensors[annotator] = []

            start_clip = int(row[12])
            end_clip = int(row[13])
            
            clip_tensors = feature_tensors[start_clip:end_clip+1] 

            #Staked tensor
            stacked_tensors = torch.cat(clip_tensors, dim=0)            
            
            #Take the max
            max_tensor = torch.max(stacked_tensors, dim=0).values
            max_tensor = max_tensor.unsqueeze(0)
            

            aggregated_tensors[annotator].append(max_tensor)
             

            

for annotator, max_tensors in aggregated_tensors.items():
    
    #directory 
    clips_dir = os.path.join(saving_dir, IMDB_KEY, annotator)
    if not os.path.exists(clips_dir): os.makedirs(clips_dir)
        
    for i, max_tensor in enumerate(max_tensors):
        tensor_path = os.path.join(clips_dir, f'{IMDB_KEY}_{i}.pt')
        torch.save(max_tensor, tensor_path) 
        
