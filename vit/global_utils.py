import torch
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
	average_precision_score

from models.vit_models import *
from models.mlp_models import *
from models.dann import DANNViT, DifferentialDANNViT
from models.cem_models.cem import ConceptEmbeddingModel, ConceptEmbeddingModelWithBackbone
from constants import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so that gpu index matches nvidia-smi index


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


def count_parameters(named_params):
	total_params = 0
	print("\nTrainable Model Parameters:")

	for name, param in named_params:
		if param.requires_grad:
			param_count = param.numel()  # Calculate the number of elements in the tensor
			total_params += param_count
			print(f"Parameter name: {name}, Size: {param.size()}, Number of parameters: {param_count}")

	print(f"\nTotal number of trainable parameters: {total_params}")


def count_all_parameters(named_params): # even non-trainable
	total_params = 0
	print("\nAll Model Parameters:")

	for name, param in named_params:
		param_count = param.numel()  # Calculate the number of elements in the tensor
		total_params += param_count
		print(f"Parameter name: {name}, Size: {param.size()}, Number of parameters: {param_count}, Requires_grad : {param.requires_grad}")

	print(f"\nTotal number of (even non-trainable) parameters: {total_params}")


def summarize_initial_weights(model, as_dataframe=True, print_topk=20):
	"""
	Summarize initial weights of a PyTorch model.

	Returns either:
	  - a pandas.DataFrame with one row per parameter tensor, plus a global row
	  - or a plain dict if as_dataframe=False
	"""
	rows = []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue

		data = param.detach().cpu().float().view(-1)

		row = {
			"name": name,
			"numel": int(data.numel()),
			"mean": float(data.mean()),
			"std": float(data.std(unbiased=False)),
			"min": float(data.min()),
			"max": float(data.max()),
			"median": float(data.median()),
			"abs_mean": float(data.abs().mean()),
		}
		rows.append(row)

	all_vals = torch.cat(
		[p.detach().cpu().float().view(-1) for p in model.parameters() if p.requires_grad]
	)
	global_row = {
		"name": "__global__",
		"numel": int(all_vals.numel()),
		"mean": float(all_vals.mean()),
		"std": float(all_vals.std(unbiased=False)),
		"min": float(all_vals.min()),
		"max": float(all_vals.max()),
		"median": float(all_vals.median()),
		"abs_mean": float(all_vals.abs().mean()),
	}
	rows.append(global_row)

	if as_dataframe:
		df = pd.DataFrame(rows)
		# Optionally print a quick summary of the largest tensors
		if print_topk is not None:
			print(df.sort_values("numel", ascending=False).head(print_topk))
		return df
	else:
		return rows


def custom_collate_fn(batch):
	visual_features = torch.stack([item[0] for item in batch])
	visual_masks = torch.stack([item[1] for item in batch])
	clip_labels = torch.tensor([item[2] for item in batch])
	one_hot_concepts = torch.stack([item[3] for item in batch])

	return visual_features, visual_masks, clip_labels, one_hot_concepts


def custom_collate_fn_with_guids(batch):
	visual_features = torch.stack([item[0] for item in batch])
	visual_masks = torch.stack([item[1] for item in batch])
	clip_labels = torch.tensor([item[2] for item in batch])
	one_hot_concepts = torch.stack([item[3] for item in batch])
	guids = np.stack([item[4] for item in batch])

	return visual_features, visual_masks, clip_labels, one_hot_concepts, guids


def custom_collate_fn_with_movie_ids(batch):
	visual_features = torch.stack([item[0] for item in batch])
	visual_masks = torch.stack([item[1] for item in batch])
	clip_labels = torch.tensor([item[2] for item in batch])
	one_hot_concepts = torch.stack([item[3] for item in batch])
	ids = torch.cat([torch.from_numpy(np.array([item[4]]).flatten())for item in batch])
	# Depending on the case, item[4] may be a single element (standard case) or an array (MIL) ;
	# hence the use of flatten to easily unify the code, and the use of np to easily use flatten()

	return visual_features, visual_masks, clip_labels, one_hot_concepts, ids


def multimodal_custom_collate_fn_with_movie_ids(batch):
	visual_features = [torch.stack([item[0][i_mod] for item in batch]) for i_mod in range(len(batch[0][0]))]
	visual_masks = [torch.stack([item[1][i_mod] for item in batch]) for i_mod in range(len(batch[0][1]))]

	clip_labels = torch.tensor([item[2] for item in batch])
	one_hot_concepts = torch.stack([item[3] for item in batch])
	ids = torch.cat([torch.from_numpy(np.array([item[4]]).flatten())for item in batch])
	# Depending on the case, item[4] may be a single element (standard case) or an array (MIL) ;
	# hence the use of flatten to easily unify the code, and the use of np to easily use flatten()

	return visual_features, visual_masks, clip_labels, one_hot_concepts, ids


def custom_collate_fn_with_guids_movie_ids(batch):
	visual_features = torch.stack([item[0] for item in batch])
	visual_masks = torch.stack([item[1] for item in batch])
	clip_labels = torch.tensor([item[2] for item in batch])
	one_hot_concepts = torch.stack([item[3] for item in batch])
	guids = np.stack([item[4] for item in batch])
	ids = torch.cat([torch.from_numpy(np.array([item[5]]).flatten())for item in batch])
	# Depending on the case, item[4] may be a single element (standard case) or an array (MIL) ;
	# hence the use of flatten to easily unify the code, and the use of np to easily use flatten()

	return visual_features, visual_masks, clip_labels, one_hot_concepts, guids, ids


def multimodal_custom_collate_fn_with_guids_movie_ids(batch):
	visual_features = [torch.stack([item[0][i_mod] for item in batch]) for i_mod in range(len(batch[0][0]))]
	visual_masks = [torch.stack([item[1][i_mod] for item in batch]) for i_mod in range(len(batch[0][1]))]

	clip_labels = torch.tensor([item[2] for item in batch])
	one_hot_concepts = torch.stack([item[3] for item in batch])
	guids = np.stack([item[4] for item in batch])
	ids = torch.cat([torch.from_numpy(np.array([item[5]]).flatten())for item in batch])
	# Depending on the case, item[4] may be a single element (standard case) or an array (MIL) ;
	# hence the use of flatten to easily unify the code, and the use of np to easily use flatten()

	return visual_features, visual_masks, clip_labels, one_hot_concepts, guids, ids


def calculate_metrics(y_true, y_pred, y_scores):
	accuracy = accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)
	f1_negative = f1_score(y_true, y_pred, pos_label=0)
	f1_macro = f1_score(y_true, y_pred, average='macro')
	f1_w = f1_score(y_true, y_pred, average='weighted')
	precision = precision_score(y_true, y_pred, zero_division=0.0)
	# there are no predicted positive for all_negative trivial baseline, hence the need of zero_division
	recall = recall_score(y_true, y_pred)
	roc_auc = roc_auc_score(y_true, y_scores)
	pr_auc = average_precision_score(y_true, y_scores)

	return accuracy, f1, f1_negative, f1_macro, f1_w, precision, recall, roc_auc, pr_auc


def calculate_tb_metrics(loss, binary_targets_np, pred_label_np, probabilities_np,
						 num_output_neurons=1):
	metrics = {}
	metrics['loss'] = loss

	if num_output_neurons > 1:
		accuracy = accuracy_score(binary_targets_np, pred_label_np)
		metrics['accuracy'] = accuracy

	else:
		# Binary metrics
		accuracy, f1, f1_negative, f1_macro, f1_w, precision, recall, roc_auc, pr_auc = calculate_metrics(
			binary_targets_np, pred_label_np, probabilities_np)
		metrics['accuracy'] = accuracy
		metrics['f1'] = f1
		metrics['f1_negative'] = f1_negative
		metrics['f1_macro'] = f1_macro
		metrics['f1_w'] = f1_w
		metrics['precision'] = precision
		metrics['recall'] = recall
		metrics['roc_auc'] = roc_auc
		metrics['pr_auc'] = pr_auc

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
						'real_n_classes': 2, 'num_output_neurons': 1}
	elif exp_name == "ENHN_S":
		classif_info = {'type_classif': 'label',
						'classif_classes': {'Easy Negative': 0, 'Hard Negative': 0, 'Sure': 1},
						'real_n_classes': 2, 'num_output_neurons': 1}
	elif exp_name == "EN_S":
		classif_info = {'type_classif': 'label',
						'classif_classes': {'Easy Negative': 0, 'Sure': 1},
						'real_n_classes': 2, 'num_output_neurons': 1}
	elif exp_name == "EN_HN_S":
		classif_info = {'type_classif': 'label',
						'classif_classes': {'Easy Negative': 0, 'Hard Negative': 1, 'Sure': 2},
						'real_n_classes': 3, 'num_output_neurons': 3}
	elif exp_name == "all_visual":
		classif_info = {'type_classif': 'several_concepts',
						'classif_classes': VISUAL_CONCEPTS,
						'real_n_classes': len(VISUAL_CONCEPTS), 'num_output_neurons': len(VISUAL_CONCEPTS)}
	elif exp_name in ALL_CONCEPTS:
		classif_info = {'type_classif': 'concept',
						'classif_classes': [exp_name],
						'real_n_classes': 2, 'num_output_neurons': 1}
	else:
		raise Exception(f"Unknwon exp_name '{exp_name}'")

	return classif_info


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

	elif classif_info["type_classif"] == 'several_concepts':
		gt_df['label_int'] = gt_df['concepts'].apply(
			lambda clip_concepts: [int(x in clip_concepts) for x in classif_classes])
		# 'label_int' is actually a list

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

	elif classif_info["type_classif"] == 'several_concepts':
		original_df['label_int'] = original_df['concepts'].apply(
			lambda clip_concepts: [int(x in clip_concepts) for x in classif_info["classif_classes"]])
		# 'label_int' is actually a list

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


def get_test_inferred_df(original_df, probabilities_np):
	# N.B. : it is important that the test dataset was not shuffled to obtain the probabilities_np
	if 'mode' in original_df.columns:
		fold_test_idxs = original_df[original_df['mode'] == 'test'].index
	else:
		fold_test_idxs = original_df[original_df['fold'] == 0].index
	if probabilities_np.shape[1] == 1: # ie. if probabilities contains the probas of the positive class only
		original_df.loc[fold_test_idxs, 'test_proba'] = probabilities_np
	else:
		for i in range(probabilities_np.shape[1]):
			original_df.loc[fold_test_idxs, f'test_proba_{i}'] = probabilities_np[:,i]

	# I don't see why saving the whole df when only the test results matter, so :
	if 'mode' in original_df.columns:
		original_df = original_df.query("mode == 'test'")
	else:
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


def standard_inference(representation_architecture, vit_model, visual_features, visual_mask, ids, clip_label, sampling_rate=0.,
					   # FOR THE CEM models
					   one_hot_concepts=None, train=False):
	# Includes ViT models and MLP model
	if representation_architecture in ["cem_vit", "cem_multimodal_vit", "backbone_cem_vit", "backbone_cem_multimodal_vit"]:
		representations, logits, c_sem = vit_model._forward(x=visual_features, mask=visual_mask, c=one_hot_concepts, train=train)
	elif representation_architecture in ["cem_mlp", "cem_complex_mlp", "cem_bigmlp", "cem_complex_bigmlp", "cem_preconcept_mlp", "cem_xclipemb"]:
		representations, logits, c_sem = vit_model._forward(x=visual_features, c=one_hot_concepts, train=train)

	else:
		c_sem = None
		if representation_architecture in ["vit", "multimodal_vit"]:
			representations, logits = vit_model(visual_features, visual_mask)
		elif representation_architecture == "differential_vit":
			representations, logits = vit_model(visual_features, visual_mask, ids)
		elif representation_architecture == "gated_differential_vit":
			representations, logits = vit_model(visual_features, visual_mask, ids)
		elif representation_architecture == "complex_gated_differential_vit":
			representations, logits = vit_model(visual_features, visual_mask, ids)
		elif representation_architecture == "attentionlike_gated_differential_vit":
			representations, logits = vit_model(visual_features, visual_mask, ids)
		elif representation_architecture == "differential_vit_negative":
			representations, logits = vit_model(visual_features, visual_mask, ids, clip_label)
		elif "differential_vit_selective_negative" in representation_architecture:
			representations, logits = vit_model(visual_features, visual_mask, ids, clip_label)
		elif "differential_vit_selective" in representation_architecture:
			representations, logits = vit_model(visual_features, visual_mask, ids)
		elif representation_architecture == "differential_vit_iterative":
			representations, logits = vit_model(visual_features, visual_mask, ids)
		elif representation_architecture == "complex_differential_vit":
			representations, logits = vit_model(visual_features, visual_mask, ids)
		elif representation_architecture == "aposteriori_differential_vit":
			representations, logits = vit_model(visual_features, visual_mask, ids)
		elif representation_architecture == "double_differential_vit":
			representations, logits = vit_model(visual_features, visual_mask, ids, clip_label)
		elif representation_architecture == "scheduled_double_differential_vit":
			representations, logits = vit_model(visual_features, visual_mask, ids, clip_label, sampling_rate)
		elif representation_architecture == "frozen_context_differential_vit":
			representations, logits = vit_model(visual_features, visual_mask, ids)

		elif representation_architecture in ["mlp", "complex_mlp", "bigmlp", "complex_bigmlp", "preconcept_mlp"]:
			representations, logits = vit_model(visual_features)
		elif representation_architecture == "differential_mlp" or representation_architecture == "differential_bigmlp":
			representations, logits = vit_model(visual_features, ids)
		elif representation_architecture == "gated_differential_mlp" or representation_architecture == "gated_differential_bigmlp":
			representations, logits = vit_model(visual_features, ids)
		elif representation_architecture == "complex_gated_differential_mlp" or representation_architecture == "complex_gated_differential_bigmlp":
			representations, logits = vit_model(visual_features, ids)
		elif representation_architecture == "attentionlike_gated_differential_mlp" or representation_architecture == "attentionlike_gated_differential_bigmlp":
			representations, logits = vit_model(visual_features, ids)
		elif representation_architecture == "differential_mlp_negative" or representation_architecture == "differential_bigmlp_negative":
			representations, logits = vit_model(visual_features, ids, clip_label)
		elif "differential_mlp_selective_negative" in representation_architecture or "differential_bigmlp_selective_negative" in representation_architecture:
			representations, logits = vit_model(visual_features, ids, clip_label)
		elif "differential_mlp_selective" in representation_architecture or "differential_bigmlp_selective" in representation_architecture:
			representations, logits = vit_model(visual_features, ids)
		#elif representation_architecture == "differential_mlp_iterative":
			#representations, logits = vit_model(visual_features, ids)
		elif representation_architecture == "complex_differential_mlp" or representation_architecture == "complex_differential_bigmlp":
			representations, logits = vit_model(visual_features, ids)
		elif representation_architecture == "aposteriori_differential_mlp" or representation_architecture == "aposteriori_differential_bigmlp":
			representations, logits = vit_model(visual_features, ids)
		elif representation_architecture == "double_differential_mlp" or representation_architecture == "double_differential_bigmlp":
			representations, logits = vit_model(visual_features, ids, clip_label)
		elif representation_architecture == "scheduled_double_differential_mlp" or representation_architecture == "scheduled_double_differential_bigmlp":
			representations, logits = vit_model(visual_features, ids, clip_label, sampling_rate)
		elif representation_architecture == "frozen_context_differential_mlp" or representation_architecture == "frozen_context_differential_bigmlp":
			representations, logits = vit_model(visual_features, ids)
		else:
			raise Exception(f"Unknown representation_architecture : '{representation_architecture}'")

	return representations, logits, c_sem


def dann_inference(representation_architecture, vit_model, visual_features, visual_mask, ids, clip_label):
	if representation_architecture[:4] == "dann":
		representations, classif_logits, domain_logits = vit_model(visual_features, visual_mask)
	elif representation_architecture[:17] == "differential_dann":
		representations, classif_logits, domain_logits = vit_model(visual_features, visual_mask, ids)
	else:
		raise Exception(f"Unknown representation_architecture : '{representation_architecture}'")
	return representations, classif_logits, domain_logits


def warm_up_lr_adaptation(global_step, warmup_steps, learning_rate, optimizer,
						  optimizer_is_attribute=False):
	# Based on optimizer_step method from the Lightning module we want to reproduce !
	# I checked that learning rates are matching with Lightning version
	if global_step < warmup_steps:
		# print("Warm-up -> reducing LR")
		lr_scale = min(1., float(global_step + 1) / warmup_steps)
		min_lr = 1e-7  # min lr threshold
		scaled_lr = max(min_lr, lr_scale * learning_rate)

		if optimizer_is_attribute: # to handle the specific case of PCGrad transformed optimizer
			for pg in optimizer._optim.param_groups:
				pg['lr'] = scaled_lr
		else:
			for pg in optimizer.param_groups:
				pg['lr'] = scaled_lr


def load_representation_model(representation_architecture,
							  global_model_path_representation,
							  device,
							  # MODEL HYPERPARAMETERS
							  feature_dim, max_seq_len, dim, depth, heads, mlp_dim,
							  pool, dim_head, dropout, emb_dropout,
							detach_representation=True,# except for CEM model, we want to detach representation
							  # To DISTINGUISH BETWEEN BINARY AND MULTICLASS CASE
							  num_output_neurons=1,
							  # SPECIFIC TO SOME MODELS ONLY
							  nb_domains=None,
							  backbone_path=None,
							  modality=None # used for CEM models in order to know the number of concepts
							  ):
	if representation_architecture == "xclipemb":
		return None

	else:

		if representation_architecture[:4] == "cem_":
			# The second part of the name indicates on which ViT architecture the CEM is built
			vit_backbone_name = representation_architecture[4:]
			vit_backbone = load_representation_model(vit_backbone_name,
													 global_model_path_representation=None,
													 device=device,
													 # MODEL HYPERPARAMETERS
													 feature_dim=feature_dim, max_seq_len=max_seq_len, dim=dim,
													 depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
													 dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
													 detach_representation=False,
													 # TO DISTINGUISH BETWEEN BINARY AND MULTICLASS CASE
													 num_output_neurons=1,
													 # the classification backbone of the ViT backbone won't be used
													 # SPECIFIC TO SOME MODELS ONLY
													 nb_domains=nb_domains,
													 backbone_path=None
													 )

			vit_model = ConceptEmbeddingModel(
				  n_concepts=len(MODALITY_TO_CONCEPTS[modality]), # Number of training-time concepts.
				  n_tasks=num_output_neurons, # Number of output labels.
				  emb_size=dim,  # We will use an embedding size of 128
				  # We do not use the loss arguments from the original implementation,
				# since the loss will be computed in the training loop
				  vit_backbone=vit_backbone, # N.B. : the classification head of the ViT backbone won't be used
					h_dim=feature_dim if vit_backbone is None else dim,
				  c2y_model=None,  # a linear layer
				return_representations=True # to be consistent with the rest of the code
				).to(device)

		elif representation_architecture[:13] == "backbone_cem_":
			vit_backbone_name = representation_architecture[13:]
			vit_backbone = load_representation_model(vit_backbone_name,
													 global_model_path_representation=backbone_path,
													 device=device,
													 # MODEL HYPERPARAMETERS
													 feature_dim=feature_dim, max_seq_len=max_seq_len, dim=dim,
													 depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
													 dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
													 detach_representation=True,
													 # TO DISTINGUISH BETWEEN BINARY AND MULTICLASS CASE
													 num_output_neurons=num_output_neurons,
													 # the classification backbone of the ViT backbone won't be used
													 # SPECIFIC TO SOME MODELS ONLY
													 nb_domains=nb_domains,
													 backbone_path=None
													 )
			vit_backbone.eval()

			vit_model = ConceptEmbeddingModelWithBackbone(
				n_concepts=len(MODALITY_TO_CONCEPTS[modality]),  # Number of training-time concepts.
				n_tasks=num_output_neurons,  # Number of output labels.
				emb_size=dim,  # We will use an embedding size of 128
				# We do not use the loss arguments from the original implementation,
				# since the loss will be computed in the training loop
				vit_backbone=vit_backbone,  # N.B. : the classification backbone of the ViT backbone won't be used
				h_dim=dim,
				c2y_model=None,  # a linear layer
				return_representations=True  # to be consistent with the rest of the code
			).to(device)

		elif representation_architecture == "representation_vit":
			vit_model = RepresentationViT(feature_dim=feature_dim, max_seq_length=max_seq_len, dim=dim,
										  depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool, dim_head=dim_head,
										  dropout=dropout, emb_dropout=emb_dropout).to(device)

		elif representation_architecture == "vit":
			vit_model = ViT(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons, dim=dim,
							depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool, dim_head=dim_head,
							dropout=dropout, emb_dropout=emb_dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture == "multimodal_vit":
			vit_model = MultiModalViT(feature_dims=feature_dim, max_seq_lengths=max_seq_len, num_classes=num_output_neurons, dim=dim,
							depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool, dim_head=dim_head,
							dropout=dropout, emb_dropout=emb_dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture == "differential_vit":
			vit_model = DifferentialViT(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons,
										dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
										dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
										return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture == "gated_differential_vit":
			vit_model = GatedDifferentialViT(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons,
										dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
										dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
										return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture == "complex_gated_differential_vit":
			vit_model = ComplexGatedDifferentialViT(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons,
										dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
										dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
										return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture == "attentionlike_gated_differential_vit":
			vit_model = AttentionLikeGatedDifferentialViT(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons,
										dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
										dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
										return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture == "differential_vit_negative":
			vit_model = DifferentialViTNegative(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons,
												dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
												dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
												return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif "differential_vit_selective_negative" in representation_architecture:
			vit_model = DifferentialViTSelectiveNegative(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons,
												dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
												dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
												return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif "differential_vit_selective" in representation_architecture:
			vit_model = DifferentialViTSelective(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons,
												dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
												dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
												return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture == "differential_vit_iterative":
			vit_model = DifferentialViTIterative(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons,
												 dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
												 dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
												 return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture == "complex_differential_vit":
			vit_model = ComplexDifferentialViT(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons,
											   dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
											   dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
											   return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture == "double_differential_vit":
			vit_model = DoubleDifferentialViT(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons,
											   dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
											   dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
											   return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture == "scheduled_double_differential_vit":
			vit_model = ScheduledDoubleDifferentialViT(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons,
											   dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
											   dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
											   return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture == "aposteriori_differential_vit":
			vit_backbone = ViT(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons, dim=dim,
							depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool, dim_head=dim_head,
							dropout=dropout, emb_dropout=emb_dropout, return_representation=True).to(device)
			vit_backbone.load_state_dict(
				torch.load(backbone_path, map_location=device, weights_only=True))
			vit_backbone.eval()  # should not even be necessary because of param.requires_grad = False

			vit_model = APosterioriDifferentialViT(backbone=vit_backbone, num_classes=num_output_neurons, dim=dim, dropout=dropout,
												   return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture == "frozen_context_differential_vit":
			vit_backbone = ViT(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons, dim=dim,
							   depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool, dim_head=dim_head,
							   dropout=dropout, emb_dropout=emb_dropout, return_representation=True).to(device)
			vit_backbone.load_state_dict(
				torch.load(backbone_path, map_location=device, weights_only=True))
			vit_backbone.eval()  # should not even be necessary because of param.requires_grad = False

			vit_model = FrozenContextDifferentialViT(feature_dim=feature_dim, max_seq_length=max_seq_len, num_classes=num_output_neurons,
										dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
										dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
										return_representation=True,
													 backbone=vit_backbone,
										  detach_representation=detach_representation).to(device)


		elif "dann" in representation_architecture:
			if representation_architecture[:17] == "differential_dann":
				vit_model = DifferentialDANNViT(feature_dim=feature_dim, max_seq_length=max_seq_len, dim=dim, depth=depth, heads=heads,
									mlp_dim=mlp_dim, pool=pool, dim_head=dim_head, dropout=dropout,
									emb_dropout=emb_dropout, num_classes=num_output_neurons, num_domains=nb_domains, classif_dropout=dropout,
									return_representation=True).to(device)
			elif representation_architecture[:4] == "dann":
				vit_model = DANNViT(feature_dim=feature_dim, max_seq_length=max_seq_len, dim=dim, depth=depth, heads=heads,
									mlp_dim=mlp_dim, pool=pool, dim_head=dim_head, dropout=dropout,
									emb_dropout=emb_dropout, num_classes=num_output_neurons, num_domains=nb_domains, classif_dropout=dropout,
									return_representation=True).to(device)
			else:
				raise Exception(f"Unknown DANN representation_architecture : '{representation_architecture}'")


		elif representation_architecture == "representation_mlp":
			vit_model = RepresentationViT(feature_dim=feature_dim, max_seq_length=max_seq_len, dim=dim,
										  depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool, dim_head=dim_head,
										  dropout=dropout, emb_dropout=emb_dropout).to(device)

		elif representation_architecture == "representation_bigmlp":
			vit_model = RepresentationViT(feature_dim=feature_dim, max_seq_length=max_seq_len, dim=dim,
										  depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool, dim_head=dim_head,
										  dropout=dropout, emb_dropout=emb_dropout).to(device)

		elif representation_architecture in ["mlp", "bigmlp"]:
			vit_model = MLP(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, representation_architecture=representation_architecture, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture == "preconcept_mlp":
			vit_model = PreConceptMLP(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture in ["differential_mlp", "differential_bigmlp"]:
			vit_model = DifferentialMLP(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, representation_architecture=representation_architecture, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture in ["gated_differential_mlp", "gated_differential_bigmlp"]:
			vit_model = GatedDifferentialMLP(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, representation_architecture=representation_architecture, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture in ["complex_gated_differential_mlp", "complex_gated_differential_bigmlp"]:
			vit_model = ComplexGatedDifferentialMLP(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, representation_architecture=representation_architecture, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture in ["attentionlike_gated_differential_mlp", "attentionlike_gated_differential_bigmlp"]:
			vit_model = AttentionLikeGatedDifferentialMLP(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, representation_architecture=representation_architecture, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture in ["differential_mlp_negative", "differential_bigmlp_negative"]:
			vit_model = DifferentialMLPNegative(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, representation_architecture=representation_architecture, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif "differential_mlp_selective_negative" in representation_architecture or "differential_bigmlp_selective_negative" in representation_architecture:
			vit_model = DifferentialMLPSelectiveNegative(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, representation_architecture=representation_architecture, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif "differential_mlp_selective" in representation_architecture or "differential_bigmlp_selective" in representation_architecture:
			vit_model = DifferentialMLPSelective(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, representation_architecture=representation_architecture, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture in ["complex_differential_mlp", "complex_differential_bigmlp"]:
			vit_model = ComplexDifferentialMLP(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, representation_architecture=representation_architecture, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture in ["double_differential_mlp", "double_differential_bigmlp"]:
			vit_model = DoubleDifferentialMLP(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, representation_architecture=representation_architecture, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture in ["scheduled_double_differential_mlp", "scheduled_double_differential_bigmlp"]:
			vit_model = ScheduledDoubleDifferentialMLP(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, representation_architecture=representation_architecture, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)

		elif representation_architecture in ["aposteriori_differential_mlp", "aposteriori_differential_bigmlp"]:
			vit_backbone = MLP(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)
			vit_backbone.load_state_dict(
				torch.load(backbone_path, map_location=device, weights_only=True))
			vit_backbone.eval()  # should not even be necessary because of param.requires_grad = False

			vit_model = APosterioriDifferentialMLP(backbone=vit_backbone, num_classes=num_output_neurons, dim=dim, dropout=dropout,
												   return_representation=True).to(device)

		elif representation_architecture in ["frozen_context_differential_mlp", "frozen_context_differential_bigmlp"]:
			vit_backbone = MLP(feature_dim=feature_dim, dim=dim, num_classes=num_output_neurons, representation_architecture=representation_architecture, dropout=dropout, return_representation=True,
										  detach_representation=detach_representation).to(device)
			vit_backbone.load_state_dict(
				torch.load(backbone_path, map_location=device, weights_only=True))
			vit_backbone.eval()  # should not even be necessary because of param.requires_grad = False

			vit_model = FrozenContextDifferentialMLP(feature_dim=feature_dim, num_classes=num_output_neurons, dim=dim, dropout=dropout,
													 return_representation=True, backbone=vit_backbone).to(device)

		else:
			raise Exception(f"Unknown representation_architecture : '{representation_architecture}'")

		if global_model_path_representation is not None:
			# it means we load a pretrained model
			vit_model.load_state_dict(
				torch.load(global_model_path_representation, map_location=device, weights_only=True), strict=False)
			# strict=False is motivated by aposteriori model : we only want to load the non-backbone weights
			vit_model.eval() # because we will only call this function as a backbone/for evaluation
			return vit_model
		else:
			# We return a randomly initialized model
			return vit_model


def get_embeddings_labels_ids_from_loader(loader, device,
										  representation_architecture, vit_model):
	# N.B. : it works even if the loader contains a single batch
	train_embeddings, train_labels, train_ids = [], [], []
	with torch.no_grad():
		for i_train, batch in enumerate(loader):
			visual_features, visual_mask, clip_label, one_hot_concepts, ids = batch
			visual_features, visual_mask, clip_label, one_hot_concepts, ids = (
				visual_features.to(device),
				visual_mask.to(device),
				clip_label.to(device),
				one_hot_concepts.to(device),
				ids.to(device))

			if representation_architecture == "representation_vit":
				representations = vit_model(visual_features, visual_mask)
			elif "vit" in representation_architecture or "mlp" in representation_architecture:
				representations, logits = standard_inference(representation_architecture=representation_architecture,
													   vit_model=vit_model,
													   visual_features=visual_features,
													   visual_mask=visual_mask, ids=ids,
													   clip_label=clip_label)
			elif "dann" in representation_architecture:
				representations, classif_logits, domain_logits = dann_inference(representation_architecture=representation_architecture,
													   vit_model=vit_model,
													   visual_features=visual_features,
													   visual_mask=visual_mask, ids=ids,
													   clip_label=clip_label)
			elif representation_architecture == "xclipemb":
				representations = torch.mean(visual_features, dim=1)

			train_embeddings.append(representations)
			train_labels.append(clip_label)
			train_ids.append(ids)

	train_embeddings = torch.cat(train_embeddings, dim=0)
	train_labels = torch.cat(train_labels, dim=0)
	train_ids = torch.cat(train_ids, dim=0)

	return train_embeddings, train_labels, train_ids
