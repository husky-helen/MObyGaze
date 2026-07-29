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

from dataset.utils.utils import listconcepts2onehot, tensor2maxseqlen, load_i3d, audiotensor2maxseqlen
from data_augmentation.data_soft_restriction import return_dataset_soft_restricted
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

class FeatureDF(Dataset):
	def __init__(self, modality, gt_csv, split="train", fold_idx=0, folds_info=None, classif_info=None,
				 max_seq_len=None, seed=None, annotator=None, balancing=True, with_guid=False, training_specified=False,
				 with_id=False, undersample=False,
				 representation_architecture=None,
				 custom_clips=False,
				 # FOR PSEUDO_LABELING
				 pseudo_labeling_dul=False,
				 pseudo_labeling_xt=False, pseudo_labels_guids=None, pseudo_labels=None, pseudo_labeling_val_to_unlabeled=False,
				 # OTHERS
				 fraction_train=None,
				 additional_train_guids=None,
				 cmsd_dataset=False, # I do not default it to True for now because I still use the "soft datasets"
				 specified_visual_feature_dir = None, # if we want to take anther than the default one
				 ):

		"""
		fold_idx :int - represents the fold number (set of distinct movies in train/val and test) to do the experiment with (in the folds info)
		fold index and fold info are related because the fold index is the key of folds info (0,1,2,3,4)
		pseudo_labeling_xt: whether this dataset corresponds to the X_t dataset defined in curriculum labeling article
		pseudo_labeling_dul: whether this dataset corresponds to the D_ul dataset defined in curriculum labeling article
		pseudo_labeling_val_to_unlabeled: if set to True, unlabeled set = val set (according to folds info) ;
		else, it corresponds to observations indicated as 'labeled'=0 (and belong to train) in the gt_csv
		(Note that, for generalization of the code, if pseudo_labeling_val_to_unlabeled, still, in the code,
		val movies are still eventually converted into train movies, and the appropriate 'labeled' column is created
		Note that all those pseudo-labeling arguments are only used if split="train"
		"""

		# If it is multimodal, we want to unify "text+visual" and "visual+test", for instance
		modalities = modality.split("+")
		modalities.sort()
		modality = "+".join(modalities)
		# We also compute several useful variables for multimodal, that still make sense for unimodal case
		nb_modalities = len(modalities)
		multimodal = nb_modalities > 1

		assert split in ["train", "validation", "test"]
		assert max_seq_len is not None

		type_classif = classif_info["type_classif"]
		assert type_classif in ["label", "concept", "several_concepts"]  # for now, these are the supported types of classification
		# As label and concepts are intrinsically different
		# (labels are mutually exclusive, concepts are not ; there is at least one label, not for concepts...)
		# we cannot just generalize the code by specifying the 'column of interest' : we need to know the type of classif

		# In the code, we assume the keys of folds_info to be integers, but because of json saving they may be strings ;
		# the following lines convert string keys into ints, so the rest of the code doesn't have to be modified
		list_keys = list(folds_info.keys())
		if isinstance(list_keys[0], str):
			for old_key in list_keys:
				folds_info[int(old_key)] = folds_info.pop(old_key)

		self.visual_feature_dir = VISUAL_FEATURE_DIR if specified_visual_feature_dir is None else specified_visual_feature_dir
		self.audio_feature_dir = AUDIO_FEATURE_DIR
		self.text_feature_dir = TEXT_FEATURE_DIR
		self.modality = modality

		self.gt_csv = gt_csv
		self.split = split
		self.fold = fold_idx
		self.type_classif = classif_info["type_classif"]  # either "label" or "concept" for now
		self.classif_classes = classif_info["classif_classes"]
		self.real_n_classes = classif_info["real_n_classes"]

		self.seed = seed
		self.annotator = annotator
		self.folds_info = deepcopy(folds_info)
		self.with_guid = with_guid
		self.with_id = with_id

		# max_seq_len may have different type depending on multimodal aspect :
		text_id = modalities.index("text") if "text" in modalities else (-1)
		visual_id = modalities.index("visual") if "visual" in modalities else (-1)
		audio_id = modalities.index("audio") if "audio" in modalities else (-1)
		self.max_seq_len_text = max_seq_len[text_id] if multimodal else max_seq_len
		self.max_seq_len_visual = max_seq_len[visual_id] if multimodal else max_seq_len
		self.max_seq_len_audio = max_seq_len[audio_id] if multimodal else max_seq_len
		# We don't care about giving wrong values to unused modalities

		self.gt_df = return_dataset_soft_restricted(self.gt_csv, modality=modality) if cmsd_dataset else (
			pd.read_csv(self.gt_csv, sep=";", converters={'concepts': pd.eval, 'start_clips': pd.eval, 'end_clips': pd.eval, 'imdb_keys': pd.eval}))
		# converters as necessary to identify concepts as a list and not a string
		# I removed index_col=0 ; shouldn't be used anyway

		if pseudo_labeling_xt or pseudo_labeling_dul:
			assert not (pseudo_labeling_xt and pseudo_labeling_dul)
			if pseudo_labeling_val_to_unlabeled:
				# We set to labeled/unlabeled the observations according to whether or not they belong to the "val set"
				self.gt_df.loc[self.gt_df.imdb_key.isin(self.folds_info[self.fold]["validation"]), 'labeled'] = 0
				self.gt_df.loc[~self.gt_df.imdb_key.isin(self.folds_info[self.fold]["validation"]), 'labeled'] = 1
				# For proper working of the following code, we switch to train the val set
				if "train" in self.folds_info[self.fold].keys():
					self.folds_info[self.fold]["train"] += self.folds_info[self.fold]["validation"]
				self.folds_info[self.fold]["validation"] = []

			# The characteristic 'labeled' of an observation can take two forms :
			# either it is a global characteristic, in which case it is indicated in the column 'labeled',
			# or it is specific to each fold and indicated in 'labeled_[i]' columns
			labeled_columns = [name for name in self.gt_df.columns if 'labeled' in name]
			labeled_columns.sort()  # so that the indices are correct
			self.labeled_columns = labeled_columns
			assert len(self.labeled_columns) > 0

		self.pseudo_labeling_xt = pseudo_labeling_xt
		self.pseudo_labeling_dul = pseudo_labeling_dul
		self.pseudo_labels_guids = pseudo_labels_guids if pseudo_labels_guids is not None else np.array([])
		self.pseudo_labels = pseudo_labels if pseudo_labels is not None else np.array([])

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
		# self.weight = self.compute_weight() # it is not used anywhere anyway

		self.id_to_positive_prop = None if self.real_n_classes > 2 else self.compute_pos_proportion()
		# used for DANN ; I want to compute this proportion before we do a possible balancing

		self.fraction_train=fraction_train
		self.additional_train_guids = additional_train_guids
		self.getsplit(balancing=balancing, training_specified=training_specified, undersample=undersample)
		# if balancing, the training set will be balanced
		# if training_specified, we will build the train set based on the specified train movies
		# (if not, train is defined as the complementary of test and validation)

		self.pos_weight = None if (self.real_n_classes > 2 and type_classif != 'several_concepts') else self.compute_pos_weight()
		# This attribute will only be used for the train split, and only in the binary case ;
		# and it will be meaningful only if balancing=False in the training

		self.representation_architecture = representation_architecture
		self.custom_clips = custom_clips

		# random seed
		setup_seed(seed)

	def get_label(self):
		"Create a label int column and keeps the original label. The labels that are not present in the dico_level are removed"

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
			# Paul : I removed it, as I don't see the point or the use in the code.
			# If I want the tru number of classes, I have some variables for that in my code
			# reverse_dico = {v: k for k, v in dico_level.items()}
			# self.gt_df['label'] = self.gt_df['label_int'].map(reverse_dico)

		elif self.type_classif == 'concept':
			positive_classes = self.classif_classes
			self.gt_df['label_int'] = self.gt_df['concepts'].apply(
				lambda clip_concepts: int(any(x in positive_classes for x in clip_concepts)))

		elif self.type_classif == 'several_concepts':
			self.gt_df['label_int'] = self.gt_df['concepts'].apply(lambda clip_concepts: [int(x in clip_concepts) for x in self.classif_classes])
			# 'label_int' is actually a list
			# N.B. :it will not be used in training anyway, since we have the one_hot_concept tensor ; both should be equal

		# self.gt_df['label_int'] = self.gt_df['label_int'].astype(int)
		# Verbosity for debugging :
		# print(f"len(self.gt_df['label_int']) : {len(self.gt_df['label_int'])}")
		# print(f"self.gt_df :\n{self.gt_df[['concepts', 'label_int']].to_string()}")

		else:
			raise Exception(f"Unknown type_classif : {self.type_classif}")

		# In case of pseudo-labeling Xt,
		# we additionally assign to their pseudo-labels the observations indicated as pseudo-labels
		if self.pseudo_labeling_xt:
			for guid, pseudo_label in zip(self.pseudo_labels_guids, self.pseudo_labels):
				self.gt_df.loc[self.gt_df['guid'] == guid, 'label_int'] = pseudo_label

	def compute_weight(self):
		# I have the impression it is not used anyway in the code ; except for computing self.weight that is not used
		sub_df = self.gt_df[~self.gt_df.imdb_key.isin(self.folds_info[self.fold]["test"] + self.folds_info[self.fold][
			"validation"])].label_int  # keep the training folds only
		return torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(sub_df), y=sub_df))

	def compute_pos_weight(self):
		if self.type_classif == 'several_concepts':
			full_labels = np.array(self.gt_df["label_int"].tolist())
			pos_weights = (np.sum(full_labels == 0, axis=0) / np.sum(full_labels == 1, axis=0)).tolist()
			return pos_weights
		else:
			label_counts = self.gt_df.label_int.value_counts()
			if len(label_counts) <= 1:
				warnings.warn("Be careful : a single class (or no class!) is present in your dataset")
				return 1.
			else:
				assert len(label_counts) == 2  # this quantity is designed for binary classification only
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
				assert len(label_counts) == 2  # this quantity is defined for binary classification only
				pos_count = label_counts.at[1]
				neg_count = label_counts.at[0]
				pos_proportion = pos_count / (pos_count + neg_count)
				id_to_positive_prop[int(id)] = pos_proportion
		return id_to_positive_prop

	def balance_classes(self, df):  # balance minority class to match majority one with random oversampling
		"""Dynamically balance the classes by oversampling the minority class."""
		if self.type_classif == 'several_concepts':
			return df
			# TODO
		else:
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
			if training_specified:
				self.gt_df = self.gt_df[self.gt_df.imdb_key.isin(self.folds_info[self.fold]["train"])]
			else:
				if 'mode' in self.gt_df.columns:
					self.gt_df = self.gt_df[self.gt_df['mode'] == "train"]
				else:
					self.gt_df = self.gt_df[~self.gt_df.imdb_key.isin(self.folds_info[self.fold]["test"] + self.folds_info[self.fold]["validation"])]

			# Additionally, in case of pseudo-labeling, we only keep the observations that are 'labeled' or used as pseudo-labels
			# If additional guids are included in train (more specifically, in D_L, in case of pseudo-labeling), we include them
			if self.pseudo_labeling_xt:
				additional_guids = self.additional_train_guids[self.fold] if self.additional_train_guids is not None else np.array([])
				if len(self.labeled_columns) == 1:
					self.gt_df = self.gt_df[np.bitwise_or(self.gt_df['labeled'] == 1, self.gt_df['guid'].isin(np.concat((self.pseudo_labels_guids, additional_guids))))]
				else:
					self.gt_df = self.gt_df[np.bitwise_or(self.gt_df[f'labeled_{self.fold}'] == 1, self.gt_df['guid'].isin(np.concat((self.pseudo_labels_guids, additional_guids))))]

			# Conversely, the unlabeled dataset only contains unlabeled observations
			if self.pseudo_labeling_dul:
				if len(self.labeled_columns) == 1:
					self.gt_df = self.gt_df[self.gt_df['labeled'] == 0]
				else:
					self.gt_df = self.gt_df[self.gt_df[f'labeled_{self.fold}'] == 0]

			# We may want to keep only a fraction of it :
			if self.fraction_train is not None:
				self.gt_df = self.gt_df.sample(frac=self.fraction_train, random_state=42)

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

	def __getitem__(self, idx):

		clip_line = self.gt_df.iloc[idx, :]  # a clip is an annotation
		clip_id = clip_line.movie_id
		clip_concepts = clip_line.concepts
		one_hot_concepts = listconcepts2onehot(clip_concepts, modality=self.modality) # original code unsqueezed it, I don't know why
		clip_label = int(clip_line.label_int) if self.type_classif != 'several_concepts' else clip_line.label_int

		if self.custom_clips:
			# For the case of (manually-created) custom clips mixing-up several movies
			# So far it is designed for the visual modality only
			# Need to create features, mask, clip_label, one_hot_concepts
			clip_imdbs = clip_line.imdb_keys
			clip_starts = clip_line.start_clips
			clip_ends = clip_line.end_clips

			list_features = [torch.load(
				os.path.join(self.visual_feature_dir, clip_imdb, clip_imdb + "_" + str(i) + ".pt")).requires_grad_(False)
							 for (clip_imdb, clip_start, clip_end) in zip(clip_imdbs, clip_starts, clip_ends) for i in range(clip_start, clip_end)
							 if os.path.exists(os.path.join(self.visual_feature_dir, clip_imdb, clip_imdb + "_" + str(i) + ".pt"))]

			visual_features = torch.cat(list_features, dim=0)
			# Each (stored) feature_clip has shape [1, 512], so now visual_features has shape [len_clip, 512]

			if "vit" in self.representation_architecture or "dann" in self.representation_architecture:
				features, mask = tensor2maxseqlen(visual_features, self.max_seq_len_visual)  # shape [160, 512]
			elif "mlp" in self.representation_architecture or "xclipemb" in self.representation_architecture:
				features = torch.mean(visual_features, dim=0)  # shape [512]
				mask = torch.Tensor([1])  # will not be used, but we still need a tensor for the collate_fn to be generalizable
			else:
				raise Exception(f"Unsupported representation_architecture : '{self.representation_architecture}'")

		else:
			clip_imdb = clip_line.imdb_key
			clip_start = clip_line.start_clip  # 16 frames clip
			clip_end = clip_line.end_clip
			# for the other modalities :
			clip_index = clip_line.clip_index
			clip_number = int(clip_index.split("_")[1])
			clip_annotator = clip_line.annotator

			# Loading the features and mask

			def load_video_features():
				# This is my function with what I have done so far
				list_features = [torch.load(
					os.path.join(self.visual_feature_dir, clip_imdb, clip_imdb + "_" + str(i) + ".pt")).requires_grad_(False)
								 for i in range(clip_start, clip_end)
								 if os.path.exists(
						os.path.join(self.visual_feature_dir, clip_imdb, clip_imdb + "_" + str(i) + ".pt"))]

				visual_features = torch.cat(list_features, dim=0)
				# Each (stored) feature_clip has shape [1, 512], so now visual_features has shape [len_clip, 512]

				if "vit" in self.representation_architecture or "dann" in self.representation_architecture:
					visual_features, visual_mask = tensor2maxseqlen(visual_features, self.max_seq_len_visual) # shape [160, 512]
				elif "mlp" in self.representation_architecture or "xclipemb" in self.representation_architecture:
					visual_features = torch.mean(visual_features, dim=0) # shape [512]
					visual_mask = torch.Tensor([1]) # will not be used, but we still need a tensor for the collate_fn to be generalizable
				else:
					raise Exception(f"Unsupported representation_architecture : '{self.representation_architecture}'")
				return visual_features, visual_mask

			def load_text_features():
				try:
					text_feature_path = os.path.join(self.text_feature_dir, clip_annotator, clip_imdb,
													 f"{clip_imdb}_{clip_number}.pt")

					if os.path.exists(text_feature_path):
						data = torch.load(text_feature_path)
						embeddings = data['embeddings'].requires_grad_(False) # shape [512, 768]
						attention_mask = data['attention_mask']

						if "vit" in self.representation_architecture or "dann" in self.representation_architecture:
							return embeddings, attention_mask
						elif "mlp" in self.representation_architecture or "xclipemb" in self.representation_architecture:
							return torch.mean(embeddings, dim=0), torch.Tensor([1])
						else:
							raise Exception(f"Unsupported representation_architecture : '{self.representation_architecture}'")
					else:
						raise ValueError("Text embeddings are missing")

				except Exception as e:
					print(
						f"Skipping index {idx}: Error loading text features - {e} - \n Annotation: {clip_imdb}, idx {clip_number} annotator {clip_annotator}")
					return self.__getitem__(idx + 1)

			def load_audio_features():
				try:
					audio_path = os.path.join(self.audio_feature_dir, clip_annotator, clip_imdb,
											  f"{clip_imdb}_{clip_number}")
					if not os.path.exists(audio_path):
						raise FileNotFoundError(f"Audio path does not exist: {audio_path}")

					audio_file_pattern = os.path.join(audio_path, f"{clip_imdb}_{clip_number}_idx_*.pt")
					audio_files = sorted(glob.glob(audio_file_pattern))
					# sorted_audio_files = sorted(os.listdir(audio_path), key=lambda x: int(x.removesuffix('.pt').split("_")[-1]))
					# audio_files = [f"{audio_path}/{audio_file}" for audio_file in sorted_audio_files]
					# We don't need that alternative way

					# So far so good : the paths of audiofiles are correct

					if not audio_files:
						raise FileNotFoundError(f"No audio files found at: {audio_path}")

					all_audio_features = []

					for file in audio_files:
						sample_feature = torch.load(file, weights_only=True)
						#['input_values'] # weights_only=True in original code ; and there was no required_rad
						# The loaded file was a BatchFeature, ie. a dict with a single key 'input_values'
						all_audio_features.append(sample_feature)

					concatenated_features = torch.cat(all_audio_features, dim=0) # shape [len_clip, 768]
					embeddings, attention_mask = audiotensor2maxseqlen(concatenated_features, self.max_seq_len_audio) # shape [160, 768]

					if "vit" in self.representation_architecture or "dann" in self.representation_architecture:
						return embeddings, attention_mask
					elif "mlp" in self.representation_architecture or "xclipemb" in self.representation_architecture:
						return torch.mean(embeddings, dim=0), torch.Tensor([1])
					else:
						raise Exception(f"Unsupported representation_architecture : '{self.representation_architecture}'")

				except Exception as e:
					print(
						f"Skipping index {idx}: Error loading audio features - {e} - \n Annotation: {clip_imdb}, idx {clip_index} annotator {clip_annotator}")
					return self.__getitem__(idx + 1)

			def load_features():
				if self.modality == "visual":
					features, mask = load_video_features()
				elif self.modality == "audio":
					features, mask = load_audio_features()
				elif self.modality == "text":
					features, mask = load_text_features()
				elif self.modality == "text+visual":
					text_features, text_mask = load_text_features()
					visual_features, visual_mask = load_video_features()
					# N.B. : We have to be careful to check the order of modalities, which is the alphabetical order
					features = [text_features, visual_features]
					mask = [text_mask, visual_mask]
				elif self.modality == "audio+text+visual":
					audio_features, audio_mask = load_audio_features()
					text_features, text_mask = load_text_features()
					visual_features, visual_mask = load_video_features()
					# N.B. : We have to be careful to check the order of modalities, which is the alphabetical order
					features = [audio_features, text_features, visual_features]
					mask = [audio_mask, text_mask, visual_mask]
				else:
					raise Exception(f"Unsupported modality : {self.modality}")
				return features, mask

			features, mask = load_features()

		if self.with_guid:
			clip_guid = clip_line.guid
			if self.with_id:
				return features, mask, clip_label, one_hot_concepts, clip_guid, clip_id
			else:
				return features, mask, clip_label, one_hot_concepts, clip_guid
		else:
			if self.with_id:
				return features, mask, clip_label, one_hot_concepts, clip_id
			else:
				return features, mask, clip_label, one_hot_concepts

	# To easily retrieve the matrix of embeddings from a dataset
	def get_extracted_features(self):
		all_features = []
		for clip_line in self.gt_df.itertuples(index=False):
			clip_imdb = clip_line.imdb_key
			clip_start = clip_line.start_clip  # 16 frames clip
			clip_end = clip_line.end_clip
			clip_guid = clip_line.guid

			list_features = [torch.load(
				os.path.join(self.visual_feature_dir, clip_imdb, clip_imdb + "_" + str(i) + ".pt")).requires_grad_(False)
							 for i in range(clip_start, clip_end)
							 if os.path.exists(os.path.join(self.visual_feature_dir, clip_imdb, clip_imdb + "_" + str(i) + ".pt"))]
			visual_features = torch.cat(list_features, dim=0)
			visual_features = torch.mean(visual_features, dim=0).cpu().numpy()
			all_features.append(visual_features)

		return np.stack(all_features, axis=0)
