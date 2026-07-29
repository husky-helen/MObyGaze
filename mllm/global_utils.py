import torch
import os
import random
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
	average_precision_score
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor, AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
	DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import copy
import re
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from PIL import Image

from constants import *


def return_dataset_soft_restricted(gt_csv, modality):
	# Unlike create_dataset_visual, HNS that do not contain visual queues are made EN instead of dropped
	# N.B. : this is the corrected version, ie. Not Sure are kept Not Sure

	# If it is multimodal, we want to unify "text+visual" and "visual+test", for instance
	modalities = modality.split("+")
	modalities.sort()

	df = pd.read_csv(gt_csv, sep=";", converters={'concepts': pd.eval, 'start_clips': pd.eval, 'end_clips': pd.eval, 'imdb_keys': pd.eval})

	visual_concepts = [MODALITY_TO_CONCEPTS[mod] for mod in modalities]
	visual_concepts = list(itertools.chain.from_iterable(visual_concepts))
	some_visual_concept = df['concepts'].apply(lambda clip_concepts: any(x in visual_concepts for x in clip_concepts))
	no_visual_concept = np.bitwise_not(some_visual_concept)

	df_visual = df[some_visual_concept]
	df_non_visual = df[no_visual_concept]
	df_non_visual.loc[df_non_visual['label'] != 'Not Sure','label'] = 'Easy Negative'
	final_df = pd.concat([df_visual, df_non_visual])

	return final_df


def prompt_from_characteristics(prompt_characteristics, modality='visual'):
	task = prompt_characteristics['exp_name']
	prompt_length = prompt_characteristics['spec']
	prompt_name = f"{task}_{prompt_length}_prompt" + (f"_{modality}" if modality != "visual" else "")
	text_prompt = open(f"{PROMPT_DIR}/{prompt_name}.txt", "r").read()
	return text_prompt


def answer_from_text_output(text_output, str_conversion):
	# First, we look for the actual template
	start = text_output.find("<answer>")
	end = text_output.find("</answer>")
	if (start != -1 and end != -1):
		answer = text_output[start + 8:end].strip().strip(".")
		answer = re.sub(r"[^a-zA-Z0-9]", "", answer)  # we clean it
		return answer
	elif (start != -1):
		answer = text_output[start + 8:].strip().strip(".")
		answer = re.sub(r"[^a-zA-Z0-9]", "", answer)  # we clean it
		return answer
	else:
		# As the template is not respected, we first simply look for the end of the output :
		output_end = text_output[(-30):].lower()
		for text_class in str_conversion.keys():
			if text_class is not None and text_class.lower() in output_end:
				return text_class
		return None


def predictions_from_answers(answers, str_conversion):
	res = []
	for answer in answers:
		if answer is None:
			res.append(None)
		else:
			pred = str_conversion[answer] if answer in str_conversion.keys() else None
			res.append(pred)
	return np.array(res)


def save_df_to_csv(df, csv_directory, csv_name):
	if not os.path.exists(csv_directory):
		os.makedirs(csv_directory)
	combined_csv_path = f"{csv_directory}/{csv_name}.csv"
	df.to_csv(combined_csv_path, sep=';', index=False)
	print(f"\nSaved successfully per-observaton results at {combined_csv_path}")


def set_gpu(gpu_index=0):
	if torch.cuda.is_available():
		torch.cuda.set_device(gpu_index)
		device = torch.device("cuda")
		print(f"using GPU N : {gpu_index}")
	else:
		device = torch.device("cpu")
	return device


def setup_seed(seed=None):
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)  # cpu
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def calculate_hard_metrics(y_true, y_pred):
	accuracy = accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)
	f1_negative = f1_score(y_true, y_pred, pos_label=0)
	f1_macro = f1_score(y_true, y_pred, average='macro')
	f1_w = f1_score(y_true, y_pred, average='weighted')
	precision = precision_score(y_true, y_pred, zero_division=0.0)
	# there are no predicted positive for all_negative trivial baseline, hence the need of zero_division
	recall = recall_score(y_true, y_pred)

	return accuracy, f1, f1_negative, f1_macro, f1_w, precision, recall


def get_hard_metrics(binary_targets_np, pred_label_np):
	metrics = {}
	# Binary metrics
	accuracy, f1, f1_negative, f1_macro, f1_w, precision, recall = calculate_hard_metrics(binary_targets_np, pred_label_np)
	metrics['accuracy'] = accuracy
	metrics['f1'] = f1
	metrics['f1_negative'] = f1_negative
	metrics['f1_macro'] = f1_macro
	metrics['f1_w'] = f1_w
	metrics['precision'] = precision
	metrics['recall'] = recall

	return metrics


# Source - https://stackoverflow.com/a
# Posted by Martin Thoma, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-08, License - CC BY-SA 4.0
def get_one_hot(targets, nb_classes):
	res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
	return res.reshape(list(targets.shape)+[nb_classes])


def classif_info_from_exp_name(exp_name):
	if exp_name == "EN_HNS":
		classif_info = {'type_classif': 'label',
						'classif_classes': {'Easy Negative': 0, 'Hard Negative': 1, 'Sure': 1},
						'real_n_classes': 2, 'num_output_neurons': 1,
						'exp_name': exp_name}
	elif exp_name == "ENHN_S":
		classif_info = {'type_classif': 'label',
						'classif_classes': {'Easy Negative': 0, 'Hard Negative': 0, 'Sure': 1},
						'real_n_classes': 2, 'num_output_neurons': 1,
						'exp_name': exp_name}
	elif exp_name == "EN_S":
		classif_info = {'type_classif': 'label',
						'classif_classes': {'Easy Negative': 0, 'Sure': 1},
						'real_n_classes': 2, 'num_output_neurons': 1,
						'exp_name': exp_name}
	elif exp_name == "EN_HN_S":
		classif_info = {'type_classif': 'label',
						'classif_classes': {'Easy Negative': 0, 'Hard Negative': 1, 'Sure': 2},
						'real_n_classes': 3, 'num_output_neurons': 3,
						'exp_name': exp_name}
	elif exp_name in ALL_CONCEPTS:
		classif_info = {'type_classif': 'concept',
						'classif_classes': [exp_name],
						'real_n_classes': 2, 'num_output_neurons': 1,
						'exp_name': exp_name}
	else:
		raise Exception(f"Unknwon exp_name '{exp_name}'")

	return classif_info


def str_conversion_from_exp_name(exp_name):
	if exp_name == "EN_HN_S":
		str_conversion = {'Normal': 0, 'Ambiguous': 1, 'Objectifying': 2, None: None}
	else:
		str_conversion = {'Normal': 0, 'Objectifying': 1, None: None}

	return str_conversion


def get_label(gt_df, classif_info):
	"""Create a label_int column and keeps the original label. 
	The labels that are not present in the dico_level are removed"""

	type_classif = classif_info["type_classif"]  # either "label" or "concept" for now
	classif_classes = classif_info["classif_classes"]

	if type_classif == 'label':
		dico_level = classif_classes

		reverse_dico = {v: k for k, v in dico_level.items()}

		# create the label_int column and filter out labels not in dico_level
		gt_df['label_int'] = gt_df['label'].map(dico_level)
		gt_df = gt_df.dropna(subset=['label_int'])

		# convert label_int to integer type
		gt_df['label_int'] = gt_df['label_int'].astype(int)

	elif type_classif == 'concept':
		positive_classes = classif_classes
		gt_df['label_int'] = gt_df['concepts'].apply(
			lambda clip_concepts: int(any(x in positive_classes for x in clip_concepts)))
	# gt_df['label_int'] = gt_df['label_int'].astype(int)
	# Verbosity for debugging :
	# print(f"len(gt_df['label_int']) : {len(gt_df['label_int'])}")
	# print(f"gt_df :\n{gt_df[['concepts', 'label_int']].to_string()}")

	else:
		raise Exception(f"Unknown type_classif : {type_classif}")
	
	return gt_df


def get_specific_df(gt_csv, classif_info, folds_info, fold_idx, annotator, restrict_to=None, return_all_folds=False):
	original_df = pd.read_csv(gt_csv, sep=';', converters={'concepts': pd.eval})
	# Selection of annotator
	if annotator == "UNION":
		pass
	elif annotator is not None:
		original_df = original_df[original_df['annotator'] == annotator]
	else:
		raise ValueError("You need to specify an annotator")

	# We only keep appropriate rows
	if classif_info["type_classif"] == "label":
		dico_level = classif_info["classif_classes"]
		valid_labels = set(dico_level.keys())
		original_df = original_df[original_df['label'].isin(valid_labels)]

		original_df['label_int'] = original_df['label'].map(dico_level)

		if original_df['label_int'].isnull().any():
			raise Exception(
				"Some labels in 'label' column could not be mapped. Check dico_level for completeness.")

	elif classif_info["type_classif"] == "concept":
		positive_classes = classif_info["classif_classes"]
		original_df['label_int'] = original_df['concepts'].apply(
			lambda clip_concepts: int(any(x in positive_classes for x in clip_concepts)))

	else:
		raise Exception(f"Unknown type_classif : {classif_info['type_classif']}")

	if return_all_folds:
		return original_df

	original_df['fold'] = -1  # init with -1
	fold_data = folds_info[fold_idx]  # Get the fold data for the current fold_idx
	test_ids = fold_data['test']
	val_ids = fold_data['validation']
	original_df.loc[original_df['imdb_key'].isin(test_ids), 'fold'] = 0  # test is fold 0
	original_df.loc[original_df['imdb_key'].isin(val_ids), 'fold'] = 1  # val id fold 1
	original_df.loc[original_df['fold'] == -1, 'fold'] = 2 # train id fold 2

	if restrict_to is not None:
		if restrict_to == "test":
			original_df = original_df.query("fold == 0")
		if restrict_to == "val":
			original_df = original_df.query("fold == 1")
		if restrict_to == "train":
			original_df = original_df.query("fold == 2")

	return original_df


def get_test_inferred_df(original_df, preds_np, original_preds_np,
						 log_txts, list_fs_label_strs=None, list_fs_guids=None):
	# N.B. : it is important that the test dataset was not shuffled to obtain the probabilities_np
	fold_test_idxs = original_df[original_df['fold'] == 0].index
	original_df['test_pred'] = pd.NA
	original_df['test_pred'] = original_df['test_pred'].astype("Int64")
	original_df.loc[fold_test_idxs, 'test_pred'] = preds_np
	# The original preds (may be None)
	original_df['original_pred'] = pd.NA
	original_df['original_pred'] = original_df['original_pred'].astype("Int64")
	original_df.loc[fold_test_idxs, 'original_pred'] = original_preds_np
	# The complete output for futher analysis
	original_df.loc[fold_test_idxs, 'output_text'] = log_txts
	if list_fs_label_strs is not None:
		# Since the following are lists of lists, we should be more careful :
		original_df['few_shot_label_strs'] = None  # or dtype object initialization
		original_df.loc[fold_test_idxs, 'few_shot_label_strs'] = pd.Series(list_fs_label_strs, index=fold_test_idxs)
		original_df['few_shot_guids'] = None  # or dtype object initialization
		original_df.loc[fold_test_idxs, 'few_shot_guids'] = pd.Series(list_fs_guids, index=fold_test_idxs)
	# There is no need to save the whole dataset when we've only computed preds for one fold :
	original_df = original_df.query('fold == 0')
	# plus we remove this costly but useless column
	if 'subtitle' in original_df.columns:
		original_df = original_df.drop(columns=['subtitle'])
	return original_df


def get_val_inferred_df(original_df, probabilities_np):
	# N.B. : it is important that the test dataset was not shuffled to obtain the probabilities_np
	if 'mode' in original_df.columns:
		fold_test_idxs = original_df[original_df['mode'] == 'val'].index
	else:
		fold_test_idxs = original_df[original_df['fold'] == 1].index
	if probabilities_np.shape[1] == 1:  # ie. if probabilities contains the probas of the positive class only
		original_df.loc[fold_test_idxs, 'test_proba'] = probabilities_np
	else:
		for i in range(probabilities_np.shape[1]):
			original_df.loc[fold_test_idxs, f'test_proba_{i}'] = probabilities_np[:, i]

	# I don't see why saving the whole df when only the test results matter, so :
	if 'mode' in original_df.columns:
		original_df = original_df.query("mode == 'val'")
	else:
		original_df = original_df.query('fold == 1')
	# plus we remove this costly but useless column
	if 'subtitle' in original_df.columns:
		original_df = original_df.drop(columns=['subtitle'])
	return original_df


def get_train_inferred_df(original_df, probabilities_np):
	# N.B. : it is important that the test dataset was not shuffled to obtain the probabilities_np
	# N.B. : Because of how this function is used in the code, unlike get_val_inferred_df and get_test_inferred_df,
	# we do not need to restrict the dataset before appending probabilities
	if probabilities_np.shape[1] == 1:  # ie. if probabilities contains the probas of the positive class only
		original_df.loc[:, 'test_proba'] = probabilities_np
	else:
		for i in range(probabilities_np.shape[1]):
			original_df.loc[:, f'test_proba_{i}'] = probabilities_np[:, i]

	# plus we remove this costly but useless column
	if 'subtitle' in original_df.columns:
		original_df = original_df.drop(columns=['subtitle'])
	return original_df


def get_unlabeled_inferred_df(original_df, probabilities_np, pseudo_labeling_val_to_unlabeled=False, folds_info=None, fold_idx=None):
	# First, if pseudo_labeling_val_to_unlabeled, we adapt the 'labeled' column in consequence
	if pseudo_labeling_val_to_unlabeled:
		# We set to labeled/unlabeled the observations according to whether or not they belong to the "val set"
		original_df.loc[original_df.imdb_key.isin(folds_info[fold_idx]["validation"]), 'labeled'] = 0
		original_df.loc[~original_df.imdb_key.isin(folds_info[fold_idx]["validation"]), 'labeled'] = 1

	# N.B. : it is important that the test dataset was not shuffled to obtain the probabilities_np
	fold_unlabeled_idxs = original_df[original_df['labeled'] == 0].index
	if probabilities_np.shape[1] == 1: # ie. if probabilities contains the probas of the positive class only
		original_df.loc[fold_unlabeled_idxs, 'test_proba'] = probabilities_np
	else:
		for i in range(probabilities_np.shape[1]):
			original_df.loc[fold_unlabeled_idxs, f'test_proba_{i}'] = probabilities_np[:,i]

	# I don't see why saving the whole df when only the test results matter, so :
	original_df = original_df.query('labeled == 0')
	# plus we remove this costly but useless column
	if 'subtitle' in original_df.columns:
		original_df = original_df.drop(columns=['subtitle'])
	return original_df


def get_shot_str(zero_shot, pooling_type=None, few_shot_annotator=None, selection_type=None,
				 total_expert_budget=None, expert_spec_str=None, few_shot_num=None,
				 few_shot_order=None,
				 preliminary_res_name=None):
	if zero_shot:
		return 'zero_shot'
	else:
		prompt_strategy = ''
		demo_pool = 'base' if pooling_type == 'base' else f"{total_expert_budget}_{few_shot_annotator}{expert_spec_str}_annotated"
		shot_str = f"{few_shot_num}_shot_{few_shot_order}_from_{demo_pool}_on_{selection_type}"
		if pooling_type == 'preliminary_few_shot':
			shot_str += f"_{preliminary_res_name}"
		return shot_str
