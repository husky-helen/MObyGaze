import torch
import os
import json
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
import warnings
from math import ceil

from dataset.dataset import setup_seed
from dataset.utils.utils import tensor2maxseqlen, listconcepts2onehot
from constants import *


# Inspired from Julie's WSL code
def process_feat(feat, nb_segment, length_segment, max_seq_len, type_representation,
				 r_shots):
	shots_annotation_dir = f'{DATASETS_DIR}/shots_annotations'
	with open(f'{shots_annotation_dir}/all_shots.json', 'r') as file:
		shots_dict = json.load(file)
	# type_representation controls how segments are processed :
	# - if "fixed_length", we cut the clip into segments of length length_segment ; elif "fixed_nb_segments",
	# we cut the clip into nb_segment instances
	# - if "mlp", we average the 16-frame clips inside of a segment into 1 ; elif "vit", we leave them as they are,
	# that is to say sequential
	if "fixed_nb_segments" in type_representation:
		# In the other cases (fixed_length, based_on_shots), because we must still guarantee a max_nb_segment constraint
		# (for batching purpose, similarly to max_seq_len in Transformers), we have to choose what to do if the
		# resulting number of segments is larger than max_nb_segment (whose role is played in the code by nb_segment)
		# We could transpose to this case fixed_nb_segments, but I guess it is better to perform truncation
		# in order to remain consistent for a given method
		final_nb_segment = nb_segment
		r = np.linspace(0, len(feat), nb_segment + 1, dtype=np.int64)
	elif "fixed_length" in type_representation:
		final_nb_segment = ceil(len(feat) / length_segment)
		r = np.arange(start=0, stop=len(feat), step=length_segment)
		r = r.tolist() + [len(feat)]
	elif "shot_based_segments" in type_representation:
		r = r_shots
		final_nb_segment = len(r) - 1
	else:
		raise Exception(f"type_representation is neither fixed_nb_segments, fixed_length nor shot_based_segments")

	final_nb_segment = min(final_nb_segment, nb_segment)
	# At the end, final_nb_segment is necessarily <= nb_segment

	if "mlp" in type_representation:
		new_feat = torch.zeros((nb_segment, feat.shape[1]))
		for i in range(final_nb_segment):
			if r[i]!=r[i+1]:
				new_feat[i,:] = torch.mean(feat[r[i]:r[i+1],:], 0)
			else:
				new_feat[i,:] = feat[r[i],:]
		# For the mask, of shape nb_segment, only the first actual final_nb_segment should be set to True
		visual_mask = torch.ones((nb_segment), dtype=torch.bool)
		visual_mask[final_nb_segment:] = torch.zeros((nb_segment - final_nb_segment), dtype=torch.bool)
	elif "vit" in type_representation:
		list_segments = []
		list_masks = []
		for i in range(final_nb_segment):
			if r[i]!=r[i+1]:
				segment_feat, segment_mask = tensor2maxseqlen(feat[r[i]:r[i+1],:], max_seq_len)
			else:
				segment_feat = feat[r[i],:].reshape(1, -1)
				segment_feat, segment_mask = tensor2maxseqlen(segment_feat, max_seq_len)
			list_segments.append(segment_feat)
			list_masks.append(segment_mask)
		for i in range(nb_segment - final_nb_segment):
			list_segments.append(torch.zeros((max_seq_len, feat.shape[1]), dtype=torch.bool))
			list_masks.append(torch.zeros((max_seq_len), dtype=torch.bool))
		new_feat = torch.stack(list_segments, dim=0)
		visual_mask = torch.stack(list_masks, dim=0)
	else:
		raise Exception(f"type_representation is neither mlp nor vit")

	return new_feat, visual_mask


# Inspired from Julie's WSL code
# I adapt it to match my standard dataset class FeatureDF
# (For now, some arguments are not really used, like with_guid, with_id, etc.)
class MILDataset(Dataset):
	def __init__(self, modality, gt_csv, split="train", fold_idx=0, folds_info=None, classif_info=None,
				 max_seq_len=None, seed=None, annotator=None, balancing=True, with_guid=False, training_specified=False,
				 with_id=False, undersample=False,
				 # MIL ARGUMENTS
				 is_normal=None, feature_dim=512, nb_segment=16, length_segment=5, type_representation=None):

		"""
		fold_idx :int - represents the fold number (set of distinct movies in train/val and test) to do the experiment with (in the folds info)
		fold index and fold info are related because the fold index is the key of folds info (0,1,2,3,4)
		"""
		assert split in ["train", "validation", "test"]
		assert is_normal is not None

		type_classif = classif_info["type_classif"]
		assert type_classif in ["label", "concept"]  # for now, these are the supported types of classification
		# As label and concepts are intrinsically different
		# (labels are mutually exclusive, concepts are not ; there is at least one label, not for concepts...)
		# we cannot just generalize the code by specifying the 'column of interest' : we need to know the type of classif

		# In the code, we assume the keys of folds_info to be integers, but because of json saving they may be strings ;
		# the following lines convert string keys into ints, so the rest of the code doesn't have to be modified
		list_keys = list(folds_info.keys())
		if isinstance(list_keys[0], str):
			for old_key in list_keys:
				folds_info[int(old_key)] = folds_info.pop(old_key)

		self.visual_feature_dir = VISUAL_FEATURE_DIR
		self.audio_feature_dir = AUDIO_FEATURE_DIR
		self.text_feature_dir = TEXT_FEATURE_DIR
		self.modality = modality

		self.gt_csv = gt_csv
		self.split = split
		self.type_classif = classif_info["type_classif"]  # either "label" or "concept" for now
		self.classif_classes = classif_info["classif_classes"]
		self.max_seq_len = max_seq_len
		self.seed = seed
		self.annotator = annotator
		self.folds_info = folds_info
		self.with_guid = with_guid
		self.with_id = with_id

		self.gt_df = pd.read_csv(self.gt_csv, sep=";", converters={'concepts': pd.eval})
		# converters as necessary to identify concepts as a list and not a string
		# I removed index_col=0 ; shouldn't be used anyway

		if self.folds_info is None:
			raise ValueError("Folds split cannot be None")

		if self.seed is None:
			raise ValueError("seed cannot be None for reproducibility purposes")

		if self.max_seq_len is None:
			raise ValueError("Max number of pt features available ")

		if self.annotator == "UNION":
			pass
		elif self.annotator is not None:
			self.gt_df = self.gt_df[self.gt_df['annotator'] == self.annotator]
		else:
			raise ValueError("You need to specify an annotator")

		self.fold = fold_idx

		self.get_label()
		# self.weight = self.compute_weight() # it is not used anywhere anyway

		self.id_to_positive_prop = self.compute_pos_proportion()
		# used for DANN ; I want to compute this proportion before we do a possible balancing

		self.getsplit(balancing=balancing, training_specified=training_specified, undersample=undersample)
		# if balancing, the training set will be balanced
		# if training_specified, we will build the train set based on the specified train movies
		# (if not, train is defined as the complementary of test and validation)

		self.pos_weight = self.compute_pos_weight()
		# This attribute will only be used for the train split ; and it will be meaningful only if it is unbalanced

		# random seed
		setup_seed(seed)

		# The only real modification compared to FeatureDF is that we restrict the dataset to positive (anomaly)
		# or negative (normal) examples for training:
		# N.B. : in the future, we may try finer experiments where for instance negative is the anomaly
		self.is_normal = is_normal
		self.feature_dim = feature_dim
		self.nb_segment = nb_segment # the number of segments in a bag/clip
		self.length_segment = length_segment # how long a segment is supposed to be
		self.type_representation = type_representation
		# type_representation controls how segments are processed :
		# - if "fixed_length", we cut the clip into segments of length length_segment ; elif "fixed_nb_segments",
		# we cut the clip into nb_segment instances
		# N.B. : even if fixed_length, we still use nb_segment to perform padding in order to proceed to batching,
		# even if clips may have different number of actual segments in this case
		# - if "mlp", we average the 16-frame clips inside of a segment into 1 ; elif "vit", we leave them as they are,
		# that is to say sequential


		if self.split == "train":
			if self.is_normal:
				self.gt_df = self.gt_df.query("label_int == 0")
			else:
				self.gt_df = self.gt_df.query("label_int == 1")

	def get_label(self):

		if self.type_classif == 'label':
			dico_level = self.classif_classes

			# create the label_int column and filter out labels not in dico_level
			self.gt_df['label_int'] = self.gt_df['label'].map(dico_level)
			self.gt_df = self.gt_df.dropna(subset=['label_int'])

			# convert label_int to integer type
			self.gt_df['label_int'] = self.gt_df['label_int'].astype(int)

			# update the 'label' column to match the filtered data
			# More precisely there are now as many labels as there are classes
			# (if the classif is e.g. ENHN_S, both EN and HN get the label HN)
			#reverse_dico = {v: k for k, v in dico_level.items()}
			#self.gt_df['label'] = self.gt_df['label_int'].map(reverse_dico)

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

	def compute_weight(self):
		# I have the impression it is not used anyway in the code ; except for computing self.weight that is not used
		sub_df = self.gt_df[~self.gt_df.imdb_key.isin(self.folds_info[self.fold]["test"] + self.folds_info[self.fold][
			"validation"])].label_int  # keep the training folds only
		return torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(sub_df), y=sub_df))

	def compute_pos_weight(self):
		label_counts = self.gt_df.label_int.value_counts()
		if len(label_counts) <= 1:
			warnings.warn("Be careful : a single class (or no class!) is present in your dataset")
			return 1.
		else:
			assert len(label_counts) == 2  # all our training code is designed for binary classification for now !!
			pos_count = label_counts.at[1]
			neg_count = label_counts.at[0]
			pos_weight = neg_count / pos_count
			#print(f"pos_weight = {neg_count} / {pos_count} = {pos_weight}   (split={self.split})")
			return pos_weight

	def compute_pos_proportion(self):
		id_to_positive_prop = dict()
		for id in self.gt_df["movie_id"].unique():
			sub_df = self.gt_df.query("movie_id == @id")
			label_counts = sub_df.label_int.value_counts()
			if len(label_counts) <= 1:
				warnings.warn("Be careful : a single class (or no class!) is present in your dataset")
				id_to_positive_prop[int(id)] = 1.
			else:
				assert len(label_counts) == 2  # all our training code is designed for binary classification for now !!
				pos_count = label_counts.at[1]
				neg_count = label_counts.at[0]
				pos_proportion = pos_count / (pos_count + neg_count)
				id_to_positive_prop[int(id)] = pos_proportion
		return id_to_positive_prop

	def balance_classes(self, df):  # balance minority class to match majority one with random oversampling
		"""Dynamically balance the classes by oversampling the minority class."""
		label_counts = df.label_int.value_counts()
		max_count = label_counts.max()

		balanced_df = pd.concat(
			[df[df.label_int == label_int].sample(n=max_count, replace=True) for label_int in label_counts.index])

		balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

		return balanced_df

	def undersample_classes(self, df): # not used at the end
		# balance majority class to match minority one with random undersampling
		list_movie_df = []
		for imdb_key in df.imdb_key.unique():
			movie_df = df[df.imdb_key == imdb_key]
			label_counts = movie_df.label_int.value_counts()
			min_count = label_counts.min()

			movie_balanced_df = pd.concat([movie_df[movie_df.label_int == label_int].sample(n=min_count, replace=False) for label_int in label_counts.index])
			list_movie_df.append(movie_balanced_df)

		balanced_df = pd.concat(list_movie_df).reset_index(drop=True)
		return balanced_df

	def getsplit(self, balancing=True, training_specified=False, undersample=False):
		if self.split == "test":
			self.gt_df = self.gt_df[self.gt_df.imdb_key.isin(self.folds_info[self.fold]["test"])]
		elif self.split == "validation":
			self.gt_df = self.gt_df[self.gt_df.imdb_key.isin(self.folds_info[self.fold]["validation"])]
		else:  # training
			if training_specified:
				self.gt_df = self.gt_df[self.gt_df.imdb_key.isin(self.folds_info[self.fold]["train"])]
			else:
				self.gt_df = self.gt_df[~self.gt_df.imdb_key.isin(
					self.folds_info[self.fold]["test"] + self.folds_info[self.fold]["validation"])]

			if balancing:
				print("balancing training set")
				self.gt_df = self.balance_classes(self.gt_df)
				label_counts = self.gt_df.label_int.value_counts()
				print(f"Count of samples in {self.split.upper()} after balancing for each label:")
				for label, count in label_counts.items():
					# label_name = list(self.dico_level.keys())[list(self.dico_level.values()).index(label)]
					print(f"{label}: {count}")

		# undersample : this can be used for any type of fold (train, val , test)
		if undersample:
			print("undersampling for balancing each movie")
			self.gt_df = self.undersample_classes(self.gt_df)

	def __len__(self):
		return self.gt_df.shape[0]

	def __getitem__(self, index):
		row = self.gt_df.iloc[index, :]
		movie_imdb = row.imdb_key
		clip_label = row.label_int
		clip_id = row.movie_id
		# I (Paul) also include one_hot concepts, in order to easily adapt the loader to existing code
		clip_concepts = row.concepts
		one_hot_concepts = torch.unsqueeze(listconcepts2onehot(clip_concepts, modality="visual"), 0)

		starting_clip = row.start_clip
		ending_clip = row.end_clip

		# TODO: as in Dataset, also handle the case with other modalities than visual

		list_features = [torch.load(
			os.path.join(self.visual_feature_dir, movie_imdb, movie_imdb + "_" + str(i) + ".pt")).requires_grad_(False)
						 for i in range(starting_clip, ending_clip)
						 if os.path.exists(
				os.path.join(self.visual_feature_dir, movie_imdb, movie_imdb + "_" + str(i) + ".pt"))]
		visual_features = torch.cat(list_features, dim=0) # shape [len_annotation_clip, 512]
		# This is similar code to FeatureDF : it seems more efficient than the original code using load_tensor_list
		# N.B. : requires_grad=False is crucial for the code to work properly
		# N.B. : originally the range in WSL code was range(starting_clip, ending_clip+1) :
		# so there was inconsistency with the code of FeatureDF
		# I chose the present range, which is the conservative one : "if part of the end clip belongs to
		# two annotations, it is not included". This is all the more meaningful in a cinematographic context with cuts.

		# We also define r_shots, that will only be used for shot_based_segments_MIL
		if "shot_based_segments" in self.type_representation:
			movie_shots = np.array(shots_dict[movie_imdb], dtype=int)
			clip_key_shots = movie_shots[np.bitwise_and(movie_shots > starting_clip, movie_shots < ending_clip)]
			clip_key_shots = [starting_clip] + clip_key_shots.tolist() + [ending_clip]
			r_shots = [key_shot - starting_clip for key_shot in clip_key_shots]
		else:
			r_shots = []

		visual_features, visual_mask = process_feat(visual_features, nb_segment=self.nb_segment, length_segment=self.length_segment,
									   max_seq_len=self.max_seq_len,
									   type_representation=self.type_representation,
													r_shots = r_shots)
		# visual_features has shape [nb_segment, 512] if "mlp", elif "vit" [nb_segment, max_len_seq, 512]
		# visual_msk has shape [nb_segment] if "mlp", elif "vit" [nb_segment, max_len_seq]

		# In the perspective of differential, we must also make a vector of shape [nb_segment] out of clip_id
		# N.B. : Dummy segments id should be set to -1 not to harm the technique
		clip_ids = - torch.ones(self.nb_segment)
		segments_mask = visual_mask if len(visual_mask.shape) == 1 else torch.any(visual_mask, dim=1)
		clip_ids[segments_mask] = clip_id

		if self.with_guid:
			clip_guid = row.guid
			if self.with_id:
				return visual_features, visual_mask, clip_label, one_hot_concepts, clip_guid, clip_ids
			else:
				return visual_features, visual_mask, clip_label, one_hot_concepts, clip_guid
		else:
			if self.with_id:
				return visual_features, visual_mask, clip_label, one_hot_concepts, clip_ids
			else:
				return visual_features, visual_mask, clip_label, one_hot_concepts
