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
import random
import glob
import warnings
import transformers
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from dataset.utils import compress_frame, sample_video, listconcepts2onehot, get_frames_of_video
from global_utils import return_dataset_soft_restricted, str_conversion_from_exp_name
from constants import *


def setup_seed(seed=None):
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)  # cpu
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


from transformers.feature_extraction_utils import BatchFeature
torch.serialization.add_safe_globals([BatchFeature]) # needed to load AST features
# torch.serialization.safe_globals([BatchFeature])

class MLLMDataset(Dataset):
	def __init__(self, gt_csv, modality="visual", fold_idx=0, folds_info=None,
				 classif_info=None, split="test", annotator=None,
				 num_frames=16, max_pixels=128 * 28 * 28, align_factor=28, strict_compress=False,
				 seed=42,
				 demonstration_dataset=None, # specifies the demonstration dataset, and also indicates whether we are performing few-shot,
				 selection_type=None, few_shot_num=None,
				 few_shot_order=None,
				 standardize=False, # whether we standardize the X-CLIP features used for similarity
				 precompute_features=False, #whether we pre-compute all features (to easily access them afterwards)
				 feature_extractor="xclip",
				 num_frames_max=100
				 ):

		"""

		"""
		# In the code, we assume the keys of folds_info to be integers, but because of json saving they may be strings ;
		# the following lines convert string keys into ints, so the rest of the code doesn't have to be modified
		list_keys = list(folds_info.keys())
		if isinstance(list_keys[0], str):
			for old_key in list_keys:
				folds_info[int(old_key)] = folds_info.pop(old_key)

		self.movie_dir = MOVIE_DIR
		self.imdb_to_movie_dir = IMDB_TO_MOVIE_DIR
		self.imdb_to_video_file = IMDB_TO_VIDEO_FILE
		self.fold = fold_idx
		self.annotator = annotator
		self.folds_info = deepcopy(folds_info)

		self.num_frames = num_frames
		self.max_pixels=max_pixels
		self.align_factor = align_factor
		self.strict_compress = strict_compress
		self.seed=seed # even tho for now there is nothing random in the dataset definition
		self.split = split
		self.modality=modality
		self.feature_extractor = feature_extractor
		self.num_frames_max = num_frames_max

		type_classif = classif_info["type_classif"]
		assert type_classif in ["label", "concept"]  # for now, these are the supported types of classification
		# As label and concepts are intrinsically different
		# (labels are mutually exclusive, concepts are not ; there is at least one label, not for concepts...)
		# we cannot just generalize the code by specifying the 'column of interest' : we need to know the type of classif
		self.type_classif = classif_info["type_classif"]  # either "label" or "concept" for now
		self.classif_classes = classif_info["classif_classes"]
		self.real_n_classes = classif_info["real_n_classes"]

		if isinstance(gt_csv, pd.DataFrame):
			self.gt_csv = "pandas_dataframe"
			self.gt_df = gt_csv
		else:
			self.gt_csv = gt_csv
			print(f"modality : {modality}")
			self.gt_df = return_dataset_soft_restricted(self.gt_csv, modality=modality) if modality is not None else (
				pd.read_csv(self.gt_csv, sep=";",
							converters={'concepts': pd.eval, 'start_clips': pd.eval, 'end_clips': pd.eval,
										'imdb_keys': pd.eval}))
		# return_dataset_soft_restricted overcomes modality conversion without creating a new .csv dataset everytime
		# converters as necessary to identify concepts as a list and not a string

		if self.folds_info is None:
			raise ValueError("Folds split cannot be None")

		if self.seed is None:
			raise ValueError("seed cannot be None for reproducibility purposes")

		if self.annotator == "UNION":
			pass
		elif self.annotator is not None:
			self.gt_df = self.gt_df[self.gt_df['annotator'] == self.annotator]
		else:
			raise ValueError("You need to specify an annotator")

		self.get_label()
		self.getsplit()
		# In this code, we simply select train, val or test

		# random seed
		setup_seed(seed)

		# few-shot
		self.few_shot = demonstration_dataset is not None
		if self.few_shot:
			self.demonstration_dataset = demonstration_dataset
			self.few_shot_order = few_shot_order

			# We will need int_conversion
			exp_name = classif_info["exp_name"]
			str_conversion = str_conversion_from_exp_name(exp_name)
			self.int_conversion = {v: k for k, v in str_conversion.items()}

			# We compute, once and for all, similarities between the current dataset and the demonstration dataset
			if standardize:
				current_features = self.get_extracted_features()
				demo_features = self.demonstration_dataset.get_extracted_features()
				scaler = StandardScaler()
				scaler.fit(np.concatenate([current_features, demo_features]))
				current_features = scaler.transform(current_features)
				demo_features = scaler.transform(demo_features)

			else:
				current_features = self.get_extracted_features()
				demo_features = self.demonstration_dataset.get_extracted_features()
			# current_features = current_features / np.linalg.norm(current_features, axis=1, keepdims=True)
			# demo_features = demo_features / np.linalg.norm(demo_features, axis=1, keepdims=True)
			# self.similarity = current_features @ demo_features.T # [nb_current, nb_demo]
			self.similarity = cosine_similarity(current_features, demo_features)
			#print(f"self.similarity ({self.similarity.shape}) :\n{self.similarity}")
			self.demo_label_ints = self.demonstration_dataset.get_label_ints()
			# We define the other few-shot related attributes
			self.selection_type = selection_type
			self.few_shot_num = few_shot_num

		self.precompute_features = precompute_features
		if self.precompute_features:
			self.all_features = self.get_extracted_features()

	def get_label(self):
		"Create a label int column and keeps the original label. The labels that are not present in the dico_level are removed"

		if self.type_classif == 'label':
			dico_level = self.classif_classes

			# create the label_int column and filter out labels not in dico_level
			self.gt_df['label_int'] = self.gt_df['label'].map(dico_level)
			self.gt_df = self.gt_df.dropna(subset=['label_int'])
			# convert label_int to integer type
			self.gt_df['label_int'] = self.gt_df['label_int'].astype(int)

		elif self.type_classif == 'concept':
			positive_classes = self.classif_classes
			self.gt_df['label_int'] = self.gt_df['concepts'].apply(
				lambda clip_concepts: int(any(x in positive_classes for x in clip_concepts)))
		# self.gt_df['label_int'] = self.gt_df['label_int'].astype(int)
		# Verbosity for debugging :
		# print(f"len(self.gt_df['label_int']) : {len(self.gt_df['label_int'])}")
		# print(f"self.gt_df :\n{self.gt_df[['concepts', 'label_int']].to_string()}")

		else:
			raise Exception(f"Unknown type_classif : {self.type_classif}")

	def getsplit(self):
		# Sometimes, for manual construction of fold, we have a column 'mode' simply indicating 'train', 'test' or 'val'
		if self.split == "test":
			if 'mode' in self.gt_df.columns:
				self.gt_df = self.gt_df[self.gt_df['mode'] == "test"]
			else:
				self.gt_df = self.gt_df[self.gt_df.imdb_key.isin(self.folds_info[self.fold]["test"])]
		elif self.split == "validation":
			if 'mode' in self.gt_df.columns:
				self.gt_df = self.gt_df[self.gt_df['mode'] == "val"]
			else:
				self.gt_df = self.gt_df[self.gt_df.imdb_key.isin(self.folds_info[self.fold]["validation"])]
		else:  # training
			if 'mode' in self.gt_df.columns:
				self.gt_df = self.gt_df[self.gt_df['mode'] == "train"]
			else:
				self.gt_df = self.gt_df[~self.gt_df.imdb_key.isin(self.folds_info[self.fold]["test"] + self.folds_info[self.fold]["validation"])]

	def __len__(self):
		return self.gt_df.shape[0]

	def __getitem__(self, idx):

		clip_line = self.gt_df.iloc[idx, :]  # a clip is an annotation
		clip_guid = clip_line.guid
		clip_concepts = clip_line.concepts
		one_hot_concepts = listconcepts2onehot(clip_concepts, modality="visual")
		clip_label = int(clip_line.label_int)
		clip_imdb = clip_line.imdb_key
		clip_start_frame = clip_line.start_frame
		clip_end_frame = clip_line.end_frame
		clip_framerate = clip_line.framerate
		clip_length_secs = int((clip_end_frame - clip_start_frame) // clip_framerate)
		clip_length_secs = max(1, clip_length_secs)
		actual_num_frames = int(min(clip_length_secs, self.num_frames_max)) if self.num_frames == "one_per_second" else self.num_frames
		clip_subtitles = clip_line.subtitle if isinstance(clip_line.subtitle, str) else ''

		query_video = f"{MOVIE_DIR}/{self.imdb_to_movie_dir[clip_imdb]}/{self.imdb_to_video_file[clip_imdb]}"
		frames = sample_video(query_video, start_frame=clip_start_frame, end_frame=clip_end_frame,
							  num_frames=actual_num_frames, max_pixels=self.max_pixels, factor=self.align_factor,
							  strict_compress=self.strict_compress)
		# frames is a np array

		if self.few_shot:
			similarities = self.similarity[idx]
			# a. We identify the few-shot demonstrations for this test sample
			if self.selection_type == 'similarity':
				few_shot_idxs = np.argsort(similarities)[::-1][:self.few_shot_num]
			elif self.selection_type == 'balanced_similarity':
				few_shot_idxs = []
				k_per_class = self.few_shot_num // self.real_n_classes
				for c in range(self.real_n_classes):
					# indices belonging to class c
					cls_idxs = np.where(self.demo_label_ints == c)[0]
					# similarities for this class
					cls_sims = similarities[cls_idxs]
					# sort descending
					top_local = np.argsort(cls_sims)[::-1][:k_per_class]
					# map back to original indices
					top_global = cls_idxs[top_local]
					few_shot_idxs.extend(top_global.tolist())
			elif self.selection_type == 'random':
				few_shot_idxs = np.random.choice(np.arange(len(similarities)), size=self.few_shot_num, replace=False)
			else:
				raise Exception(f"selection_type {self.selection_type} not supported for now.")
			# b. We return the embedding and label str of the few-shot demonstrations
			few_shot_frames = []
			few_shot_label_str = []
			few_shot_concept = []
			few_shot_guid = []
			for few_shot_idx in few_shot_idxs:
				demo_line = self.demonstration_dataset.gt_df.iloc[few_shot_idx, :]  # a clip is an annotation
				demo_concepts = demo_line.concepts
				demo_guid = demo_line.guid
				demo_one_hot_concepts = listconcepts2onehot(demo_concepts, modality="visual")
				demo_label_str = demo_line.label_str
				demo_imdb = demo_line.imdb_key
				demo_start_frame = demo_line.start_frame
				demo_end_frame = demo_line.end_frame
				demo_video = f"{MOVIE_DIR}/{self.imdb_to_movie_dir[demo_imdb]}/{self.imdb_to_video_file[demo_imdb]}"
				demo_frames = sample_video(demo_video, start_frame=demo_start_frame, end_frame=demo_end_frame,
				                      num_frames=self.num_frames, max_pixels=self.max_pixels, factor=self.align_factor,
				                      strict_compress=self.strict_compress)
				few_shot_frames.append(demo_frames)
				few_shot_label_str.append(demo_label_str)
				few_shot_concept.append(demo_concepts)
				few_shot_guid.append(demo_guid)

			# We reorder the few-shot examples based on self.few_shot_order
			clip_label_str = self.int_conversion[clip_label]
			if self.few_shot_order == "ground_truth_first":
				new_order = (
						[i for i, x in enumerate(few_shot_label_str) if x == clip_label_str] +
						[i for i, x in enumerate(few_shot_label_str) if x != clip_label_str]
				)
			elif self.few_shot_order == "ground_truth_last":
				new_order = (
						[i for i, x in enumerate(few_shot_label_str) if x != clip_label_str] +
						[i for i, x in enumerate(few_shot_label_str) if x == clip_label_str]
				)
			elif self.few_shot_order == "random":
				new_order = np.random.permutation(len(few_shot_label_str)).tolist()
			else:
				raise Exception(f"few_shot_order {self.few_shot_order} not supported.")

			# Reorder all lists
			def apply_order(lst):
				return [lst[i] for i in new_order]

			few_shot_label_str = apply_order(few_shot_label_str)
			few_shot_frames = apply_order(few_shot_frames)
			few_shot_concept = apply_order(few_shot_concept)
			few_shot_guid = apply_order(few_shot_guid)

			return frames, clip_label, one_hot_concepts, clip_guid, few_shot_frames, few_shot_label_str, few_shot_concept, few_shot_guid
		else:
			return frames, clip_label, one_hot_concepts, clip_guid, clip_subtitles

	def get_standardized_features(self, num_frames, max_pixels=None, factor=None):
		# In order to load all frames while enforcing compression,
		# independently of standard compression defined by attributes
		# N.B. : But those are RGB features, which are irrelevant for computing similarity
		all_features = []
		for clip_line in self.gt_df.itertuples(index=False):
			clip_imdb = clip_line.imdb_key
			clip_start_frame = clip_line.start_frame
			clip_end_frame = clip_line.end_frame
			clip_framerate = clip_line.framerate
			clip_length_secs = (clip_end_frame - clip_start_frame) // clip_framerate
			actual_num_frames = int(min(clip_length_secs, self.num_frames_max)) if num_frames == "one_per_second" else num_frames
			query_video = f"{MOVIE_DIR}/{self.imdb_to_movie_dir[clip_imdb]}/{self.imdb_to_video_file[clip_imdb]}"
			frames = sample_video(query_video, start_frame=clip_start_frame, end_frame=clip_end_frame,
			                      num_frames=actual_num_frames, max_pixels=max_pixels, factor=factor,
			                      strict_compress=True).reshape(-1)
			all_features.append(frames)
		return np.stack(all_features, axis=0)

	def get_extracted_features(self):
		all_features = []
		for clip_line in self.gt_df.itertuples(index=False):
			clip_imdb = clip_line.imdb_key
			clip_start = clip_line.start_clip  # 16 frames clip
			clip_end = clip_line.end_clip
			clip_guid = clip_line.guid

			if "xclip" in self.feature_extractor:
				visual_feature_dir = VISUAL_FEATURE_DIR

				list_features = [torch.load(
					os.path.join(visual_feature_dir, clip_imdb, clip_imdb + "_" + str(i) + ".pt")).requires_grad_(False)
								 for i in range(clip_start, clip_end)
								 if os.path.exists(os.path.join(visual_feature_dir, clip_imdb, clip_imdb + "_" + str(i) + ".pt"))]
				visual_features = torch.cat(list_features, dim=0)
				visual_features = torch.mean(visual_features, dim=0).cpu().numpy()
				all_features.append(visual_features)

			elif self.feature_extractor == "vit":
				feature = np.load(f"{VIT_FEATURE_DIR}/guid_{clip_guid}.npy")
				all_features.append(feature)

		return np.stack(all_features, axis=0)

	def get_filtered_features(self, column_name, accepted_values):
		assert self.precompute_features
		return self.all_features[self.gt_df[column_name].isin(accepted_values)]

	def get_label_ints(self):
		return self.gt_df["label_int"].to_numpy()

	def get_field(self, field):
		return self.gt_df[field].to_numpy()

	def get_clip_frames(self, idx, num_frames, max_pixels=None, factor=None):

		clip_line = self.gt_df.iloc[idx, :]  # a clip is an annotation
		clip_guid = clip_line.guid
		clip_concepts = clip_line.concepts
		one_hot_concepts = listconcepts2onehot(clip_concepts, modality="visual")
		clip_label_str = clip_line.label
		clip_title = clip_line.movie_title
		clip_imdb = clip_line.imdb_key
		clip_start_frame = clip_line.start_frame
		clip_end_frame = clip_line.end_frame
		clip_framerate = clip_line.framerate
		clip_length_secs = (clip_end_frame - clip_start_frame) // clip_framerate
		actual_num_frames = int(min(clip_length_secs, self.num_frames_max)) if num_frames == "one_per_second" else num_frames

		query_video = f"{MOVIE_DIR}/{self.imdb_to_movie_dir[clip_imdb]}/{self.imdb_to_video_file[clip_imdb]}"
		frames = get_frames_of_video(query_video, start_frame=clip_start_frame, end_frame=clip_end_frame,
									 num_frames=actual_num_frames, max_pixels=max_pixels, factor=factor)
		# frames is a list of np arrays
		return clip_title, clip_guid, clip_label_str, frames

	def get_clip_info(self, idx):
		clip_line = self.gt_df.iloc[idx, :]  # a clip is an annotation
		clip_movie_id = clip_line.movie_id
		clip_index = clip_line.clip_index
		return clip_movie_id, clip_index
