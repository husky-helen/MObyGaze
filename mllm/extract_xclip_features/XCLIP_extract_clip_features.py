import torch
import numpy as np
import json
import os
import pandas as pd
from decord import VideoReader, cpu
import numpy as np
import cv2
from math import ceil
import time
import gc

from extract_xclip_features.utils_models import load_model
from constants import *


imdb_to_movie_dir = IMDB_TO_MOVIE_DIR
imdb_to_video_file = IMDB_TO_VIDEO_FILE

def xclip_extract_clip_features(imdbs, model_name):
	processor, model = load_model(model_name=model_name)
	model.eval()

	print("PROCESSOR")
	print(processor)
	print("MODEL")
	print(model)
	print(f"\nmodel.config:\n{model.config}")
	if processor is None or model is None:
		print(f"Failed to load the model with name {model_name}.")
		print("Please check if the model name contains 'xclip' or 'vivit' and if the model is available.")
		return

	for imdb in imdbs:
		print(f"\n\n## {imdb}\n")
		video_name = f"{MOVIE_DIR}/{imdb_to_movie_dir[imdb]}/{imdb_to_video_file[imdb]}"
		videoreader = VideoReader(video_name, num_threads=1, ctx=cpu(0))
		videoreader.seek(0)
		video_length = len(videoreader) # does correspond to the number of frames, and matches with int(cv2.VideoCapture(video_name).get(cv2.CAP_PROP_FRAME_COUNT))

		nb_clips = video_length // 16 # number of "complete clips" (incomplete clips would not be supported by the model)
		print(f"nb_clips: {nb_clips}")

		with torch.no_grad():

			for clip in range(nb_clips):
				#print(f"\nclip {clip}")

				indices_ = np.arange(clip*16, (clip+1)*16, dtype=int)
				#print(f"indices_: {indices_}")

				video_frames = videoreader.get_batch(indices_).asnumpy()
				target_size = (224,224)
				video = np.array([cv2.resize(frame, target_size) for frame in video_frames]) # [16, 224, 224, 3]
				#print(f"video.shape: {video.shape}")
				#print("LOADED VIDEO")
				#print(f"list(video)[0] ({list(video)[0].shape}) :\n{list(video)[0]}")
				inputs = processor(videos=list(video), return_tensors="pt") # pixel_values : [1, 16, 3, 224, 224]
				#print(f"inputs :\n{inputs['pixel_values'].shape}")

				# get features with get_video_features
				video_features = model.get_video_features(**inputs)
				#print(f"video_features :\n{video_features.shape}")

				# save features
				model_str = model_name.split('/')[-1]
				saving_path = f"{FEATURE_DIR}/{model_str}_features"
				total_saving_path = os.path.join(saving_path, imdb, f"{imdb}_{clip}.pt")
				#print(f"total_saving_path: {total_saving_path}")

				if not os.path.exists(os.path.join(saving_path, imdb)):
					os.makedirs(os.path.join(saving_path, imdb))
				torch.save(video_features.detach().cpu(), total_saving_path)

		del videoreader
		gc.collect()

	print(f"\nFINISHED !!")


if __name__ == "__main__":
	#the_help_imdb = "tt1454029"
	#the_help_clips = np.arange(2099, 2155).tolist() + np.arange(7251, 7437).tolist()

	#indiana_jones_imdb = "tt0097576"
	#indiana_jones_clips = np.arange(3959, 3987).tolist()

	#titanic_imdb = "tt0120338"
	#titanic_clips = np.arange(14237, 14258).tolist()

	#all_imdbs = [the_help_imdb, indiana_jones_imdb, titanic_imdb]
	#all_clips = [the_help_clips, indiana_jones_clips, titanic_clips]

	model_name = "microsoft/xclip-base-patch32-16-frames" # We don't have much choice in models supporting 16-frames inputs

	start = time.time()
	xclip_extract_clip_features(imdbs=ALL_MOVIES,
								model_name=model_name)
	end = time.time()
	print(f"\nTotal time for all movies: {round((end - start) / 60, 2)} minutes")
