import os
import pandas as pd
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE", device)
wav2vec_model.to(device)

def extract_and_save_features(row, base_dir, output_dir):
    clip_number = row['speech_index'].split('_')[1]
    file_name = f"{row['imdb_key']}_{clip_number}_{row['annotator']}.wav"
    file_path = os.path.join(base_dir, row['imdb_key'], file_name)
    
    if not os.path.exists(file_path):
        print(f"Audio file {file_path} not found")
        return
    
    try:
        signal, sample_rate = torchaudio.load(file_path)
    except Exception as e:
        print("Failed to decode", file_name)
        return
    
    if signal.numel() == 0:
        print(f"Empty audio file {file_name}")
        return 
    
    if sample_rate != feature_extractor.sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=feature_extractor.sampling_rate)
        signal = resampler(signal)
    
       
    signal = signal.squeeze(0) #remove channel dim when batch size is 1 
    inputs = feature_extractor(signal, sampling_rate=feature_extractor.sampling_rate, do_normalize = True, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
        print("last hidden states")
        print(outputs.last_hidden_state.shape)


    pooled_output = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu()
    
    print(f"Shape of the extracted features: {pooled_output.shape}")
    imdb_dir = os.path.join(output_dir, row['imdb_key'])
    os.makedirs(imdb_dir, exist_ok=True)

    feature_path = os.path.join(imdb_dir, f"{row['imdb_key']}_{clip_number}_{row['annotator']}.pt")
    torch.save(pooled_output, feature_path)
    print(f"Saved features to {feature_path}")
        
    return

csv_file = '/datadir/data/annotator_1_audio.csv'
base_dir = '/datadir/segments/annotator_1'
output_dir = '/datadir/annotator_1'

os.makedirs(output_dir, exist_ok=True)

data = pd.read_csv(csv_file, delimiter=';')

for idx, row in data.iterrows():
    extract_and_save_features(row, base_dir, output_dir)
    torch.cuda.empty_cache() 