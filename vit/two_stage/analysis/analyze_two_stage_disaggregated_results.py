import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
	average_precision_score
import re
import json
import numpy as np
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import pearsonr, spearmanr
import tikzplotlib

from two_stage.two_stage_train import get_repr_strs
from constants import *
from global_utils import get_label, classif_info_from_exp_name, get_one_hot

pd.options.mode.copy_on_write = True


def get_prediction_df(exp_name, train_annotator, test_annotator, seed,
					  # TYPE OF REPRESENTATION TRAINING
					  representation_architecture, representation_loss, batch_size,
					  # CONTRASTIVE ARGUMENTS
					  global_contrastive=None, contrastive_type=None, weighted_contrastive=None, temperature=None,
					  # TYPE OF CLASSIFICATION TRAINING
					  type_classif=None, batch_size_class=None,
					  # DIFFERENTIAL TRAINING
					  nb_movies_per_batch=None, nb_clips_per_movie=None,
					  # MIL
					  nb_segment=None, length_segment=None,
					  target_model=None,
					  # OTHERS
					  train_annotator_backbone=None, representation_backbone=None,
					  full_model=False,
					  # FOR DETAILED DIRECTORY
						modality=None, pool='cls-pooling', spec_str="_two_stage", dataset_name=None,
					  # PSEUDO-LABELING AND CONNECTED SETTINGS
					  train_on_target=False, pseudo_labeling=False,
					  confidence_based_pseudo_labeling=False, budget=None,
					  # ANALYSIS PARAMETERS
					  variable_threshold=False,
					  new_classif_info=None,
					  inference_set='test', # can be 'test', 'val', 'unlabeled'
					  suffix='', # can be '' or f"_cycle_{training_cycle_id}"
					  test_on_custom_dataset=None, # whether we want the custom results instead of traditional test results
					  custom_annotator='UNION',
					  # FOR ELISA EXPERIMENT
					  fold_dataset=False, # whether there is one dataset per fold (possible when constructed manually)
					  only_one_fold=False, # for instance for training dynamics
					  # OTHERS
					fraction_train=None,
					  randint_train=True,
					  **kwargs
					  ):
	repr_str, type_representation = get_repr_strs(representation_architecture, representation_loss,
												  global_contrastive, contrastive_type, weighted_contrastive,
												  temperature,
												  nb_movies_per_batch, nb_clips_per_movie,
												  nb_segment, length_segment,
												  batch_size, train_annotator_backbone, representation_backbone,
												  target_model,
												  train_on_target, pseudo_labeling,
												  confidence_based_pseudo_labeling, budget,
												  fraction_train,
												  randint_train
												  )
	type_classif_str = type_classif if type_classif != "mlp" else f"{type_classif}_bs_{batch_size_class}"

	total_df = None
	# Will contain each line of the final df

	# The option fold_dataset is useful for an experiment from Elisa, where we manually construct the datasets,
	# which results in one dataset per fold
	if fold_dataset:
		folds = [int(dataset_name.split('_')[-1])]
	else:
		if only_one_fold:
			# In practice, the only case in which we don't have 5 folds is the training dynamics train_on_target case
			folds = [0]
		else:
			folds = np.arange(5)

	for fold_idx in folds:
		if not full_model:
			# Two-stage
			model_name = f'{type_classif_str}_{repr_str}'
		else:
			# Full model
			model_name = f'{repr_str}_full_model'

		# As we have run no actual "random" experiment (ie. output 1 with a probability of 0.5),
		# we generate artificial random results here.
		# In order to load the correct test set, we load inference results for the standard ViT architecture
		# (as we should hae results in all possible settings with this architecture) and will manually replace
		# its inference predictions by random ones
		if repr_str == "random":
			csv_directory = f"{PROJECT_DIR}/results_two_stage/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/vit_CE_bs_{batch_size}_full_model/train_{train_annotator}"
			if not os.path.exists(csv_directory):
				csv_directory = f"{PROJECT_DIR}/results_two_stage/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/multimodal_vit_CE_bs_{batch_size}_full_model/train_{train_annotator}"
			if not os.path.exists(csv_directory):
				raise Exception(f"There is no ViT experiments run with these experiments "
								f"(to use as a backbone for constructing random results)")
			combined_csv_path = f"{csv_directory}/{inference_set}_results_seed_{seed}_{test_annotator}_{exp_name}_fold_{fold_idx}{suffix}.csv"
			# If we are interested in the result of a random classification,
			# we have necessarily run a vit (or multimodal_vit) experiment on it ; beware the batch size (as it changed across modalities)
		else:
			if test_on_custom_dataset is not None:
				csv_directory = f"{PROJECT_DIR}/results_two_stage/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{repr_str}_full_model/train_{train_annotator}"
				name_custom_dataset = (test_on_custom_dataset.split('/')[-1]).split('.')[0]
				combined_csv_path = f"{csv_directory}/{name_custom_dataset}_results_seed_{seed}_{custom_annotator}_{exp_name}_fold_{fold_idx}.csv" #N.B. : we removed {pseudo_labeling_str} from the result name for convenience
			else:
				csv_directory = f"{PROJECT_DIR}/results_two_stage/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{model_name}/train_{train_annotator}"
				combined_csv_path = f"{csv_directory}/{inference_set}_results_seed_{seed}_{test_annotator}_{exp_name}_fold_{fold_idx}{suffix}.csv"

		# We load the df for this fold
		df = pd.read_csv(combined_csv_path, sep=";", converters={'concepts': pd.eval})
		# There is no need to restrict the df ; restriction to appropriate fold and test annotators
		# should have already been handled during saving

		# In order for the real fold to be accessible in the future
		df['fold'] = fold_idx

		# Because we may want to test results on different labels that the classification it was originally tested for,
		# we may want to change the label_int column
		if new_classif_info is not None:
			df = df.drop(columns=['label_int'])
			df = get_label(df, new_classif_info)

		column_names = list(df.columns)
		test_proba_columns = [name for name in column_names if 'test_proba' in name]
		test_proba_columns.sort() # so that the indices are correct
		# The test proba columns is either ['test_proba'] for binary classification,
		# and it indicates the output probability of the model, that is the probability of the true class,
		# or it is ['test_proba_0', 'test_proba_1', 'test_proba_2', ...] for multiclass classification

		if len(test_proba_columns) == 0:
			raise ValueError("CSV must contain 'test_proba[...]' columns.")

		elif len(test_proba_columns) == 1:
			# For the random baseline, we generate an artificial test_proba column = 0.5 (for soft performance metrics)
			# and a random predictions column (for hard metrics)
			if repr_str == "random":
				df.loc[:, 'test_proba'] = 1/2
				# Random (reproducible) predictions
				np.random.seed(seed)
				n = len(df)
				df.loc[:, 'predictions'] = np.random.randint(2, size=n)
			# For non-random, we convert probabilities to binary predictions using the threshold
			else:
				if variable_threshold:
					threshold = df['optimal_threshold'].iat[0]
					#print(f"optimal threshold : {threshold}")
				else:
					threshold = 0.5
				df.loc[:, 'threshold'] = threshold
				df['predictions'] = (df['test_proba'] >= threshold).astype(int)
			# We also create the confidence column
			df['confidence'] = df['label_int'] * df['test_proba'] + (1 - df['label_int']) * (1 - df['test_proba'])
			df['pseudo_confidence'] = df['confidence'].apply(lambda x: max(x, 1-x))

		else: # multiclass
			df.loc[:, 'threshold'] = np.nan
			# We generate an artificial test_proba column for the random baseline :
			if repr_str == "random":
				nb_classes = len(test_proba_columns)
				df.loc[:, test_proba_columns] = 1 / nb_classes
				# If we want random predictions, but with sure probabilities :
				#np.random.seed(seed)
				#n = len(df)
				#random_classification = np.random.randint(nb_classes, size=n)
				#random_one_hot = get_one_hot(targets = random_classification, nb_classes=nb_classes)
				#df.loc[:, test_proba_columns] = random_one_hot
			# Convert probabilities to binary predictions using the threshold
			classif_info = classif_info_from_exp_name(exp_name)
			if classif_info['type_classif'] == 'several_concepts':
				renaming_dict = {f"test_proba_{i_concept}": f"test_proba_{concept}"
								 for i_concept, concept in enumerate(classif_info['classif_classes'])}
				df = df.rename(columns=renaming_dict)
				for i_concept, concept in enumerate(classif_info['classif_classes']):
					df[f'predictions_{concept}'] = (df[f'test_proba_{concept}'] >= 0.5).astype(int)
					df[f'label_int_{concept}'] = df['concepts'].apply(lambda clip_concepts: int(concept in clip_concepts))
			else:
				df['predictions'] = np.argmax(df[test_proba_columns])
				# We also create the confidence column
				def get_confidence(line):
					return line[f"test_proba_{line['label_int']}"]
				df['confidence'] = df.apply(get_confidence, axis=1)
				df['pseudo_confidence'] = np.max(df[test_proba_columns])

		# We aggregate the folds dfs
		total_df = df if total_df is None else pd.concat([total_df, df], ignore_index=True)

	if 'subtitle' in total_df.columns:
		total_df = total_df.drop(columns=['subtitle'])

	return total_df


def calculate_metrics(y_true, y_pred, y_scores):
	num_output_neurons = y_scores.shape[1]

	if num_output_neurons == 1:

		# For computing "negative metrics", we need to define the negative counterpart
		y_scores_neg = 1 - y_scores

		# Accuracy
		accuracy = accuracy_score(y_true, y_pred)
		# F1
		f1 = f1_score(y_true, y_pred, zero_division=0.)
		f1_negative = f1_score(y_true, y_pred, pos_label=0, zero_division=0.)
		f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0.)
		f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0.)
		# Precision and recall (we need to handle the zero_division for trivail baselines, where there can be no predicted positive)
		precision = precision_score(y_true, y_pred, zero_division=0.0)
		precision_negative = precision_score(y_true, y_pred, zero_division=0.0, pos_label=0)
		recall = recall_score(y_true, y_pred, zero_division=0.0)
		recall_negative = recall_score(y_true, y_pred, zero_division=0.0, pos_label=0)

		true_positive_prop = np.count_nonzero(y_true) / len(y_true)
		pred_positive_prop = np.count_nonzero(y_pred) / len(y_pred)

		# AUC scores
		try:
			# PR AUC
			negative_pr_auc = average_precision_score(y_true, y_scores_neg, pos_label=0)
			positive_pr_auc = average_precision_score(y_true, y_scores, pos_label=1)
			auc_pr_macro = 0.5 * positive_pr_auc + 0.5 * negative_pr_auc
			# ROC AUC
			roc_auc = roc_auc_score(y_true, y_scores)
		except:
			negative_pr_auc = np.nan
			positive_pr_auc = np.nan
			auc_pr_macro = np.nan
			roc_auc = np.nan
			print(f"Warning : Only one class present in y_true. ROC AUC score is not defined in that case.")

		binary_classification_metrics = {
			"accuracy": round(accuracy, 3),
				"f1_score": round(f1, 3),
				"f1_negative": round(f1_negative, 3),
				"f1_macro": round(f1_macro, 3),
				"f1_weighted": round(f1_w, 3),
				"precision": round(precision, 3),
				"precision_negative": round(precision_negative, 3),
				"recall": round(recall, 3),
				"recall_negative": round(recall_negative, 3),
				"roc_auc": round(roc_auc, 3),
				"negative_pr_auc": round(negative_pr_auc, 3),
				"positive_pr_auc": round(positive_pr_auc, 3),
				"auc_pr_macro": round(auc_pr_macro, 3),
			"true_positive_prop": round(true_positive_prop, 3),
			"pred_positive_prop": round(pred_positive_prop, 3),
										 }
		return binary_classification_metrics

	else:
		y_scores = y_scores / y_scores.sum(axis=1, keepdims=True) # because roc_auc_score needs strict sum to 1.

		accuracy = accuracy_score(y_true, y_pred)
		auc_pr_macro = average_precision_score(y_true, y_scores, average='macro')
		roc_auc = roc_auc_score(y_true, y_scores, multi_class='ovr')

		multiclass_classification_metrics = {"accuracy": round(accuracy, 3),
											 "roc_auc": round(roc_auc, 3),
											 "auc_pr_macro": round(auc_pr_macro, 3)
											 }
		return multiclass_classification_metrics


def calculate_metrics_on_df(df, model_name, task, dataset_name, seed, type_representation, representation_architecture, representation_loss,
							modality, # the features the model is trained on, to make the distinction between vit/text and vit/visual
							spec_str,
							train_annotator=None, test_annotator=None, per_label=False, per_movie=False, detailed=True,
							with_concept_props=False, fold_dataset=False): # it is per fold anyway ; detailed=False averages all lines

	# constants
	ce_loss = representation_loss == "CE"
	differential = "differential" in representation_architecture
	representation_components = representation_architecture.split("_")
	# Now the core architecture : must be vit, mlp, dann or xclipemb, or random
	try:
		id_core = representation_components.index("vit")
	except:
		try:
			id_core = representation_components.index("mlp")
		except:
			try:
				id_core = representation_components.index("dann")
			except:
				try:
					id_core = representation_components.index("bigmlp")
				except:
					try:
						id_core = representation_components.index("xclipemb")
					except:
						id_core = -1
	core_architecture = representation_components[id_core] if id_core != -1 else "random"
	del representation_components[id_core]
	variant = "_".join(representation_components)
	variant = variant if variant != '' else 'standard'

	column_names = list(df.columns)
	test_proba_columns = [name for name in column_names if 'test_proba' in name]
	test_proba_columns.sort()  # so that the indices are correct
	# The test proba columns is either ['test_proba'] for binary classification,
	# and it indicates the output probability of the model, that is the probability of the true class,
	# or it is ['test_proba_0', 'test_proba_1', 'test_proba_2', ...] for multiclass classification

	if test_annotator == "UNION":
		pass
	elif test_annotator is not None:
		df = df[df['annotator'] == test_annotator]

	res = dict()

	# The option fold_dataset is useful for an experiment from Elisa, where we manually construct the datasets,
	# which results in one dataset per fold
	if fold_dataset:
		folds = [int(dataset_name.split('_')[-1])]
	else:
		if "train_on_target" in spec_str:
			# In practice, the only case in which we don't have 5 folds is the train_on_target case
			folds = [0]
		else:
			folds = np.arange(5).tolist()

	discriminators = ['fold']
	discriminators_values = [folds]
	if per_label:
		discriminators.append('label_int')
		discriminators_values.append(list(df['label_int'].unique()))
	if per_movie:
		discriminators.append('movie_title')
		discriminators_values.append(list(df['movie_title'].unique()))

	for combination in product(*tuple(discriminators_values)):
		query_combination = ""
		str_combination = ""
		for n, v in zip(discriminators, combination):
			str_v = f"'{v}'" if n == "movie_title" else v
			query_combination += f" and {n} == {str_v}"
			str_combination += f"{v}/"
		query_combination = query_combination[5:]
		combination_df = df.query(query_combination)

		if len(combination_df) > 0:
			# As the combination df is included in a  fold, the optimal threshold (if there is one) is the same across the combination_df
			optimal_threshold = combination_df['optimal_threshold'].iat[0] if 'optimal_threshold' in combination_df.columns else None
			threshold = combination_df['threshold'].iat[0] if 'threshold' in combination_df.columns else None

			res[str_combination] = {
				"exp_name": task,
				"spec_str": spec_str, # may be useful to make group-by's for experiments which differ only in spec_str
				"dataset_name": dataset_name,
				"model": model_name,
				"modality":modality,
				"representation_architecture": representation_architecture,
				"representation_loss": representation_loss,
				"type_representation": type_representation,
				"train_annotator": train_annotator,
				"test_annotator": test_annotator,
				"seed": seed,
				"nb_obs": len(combination_df),
				"optimal_threshold": optimal_threshold,
				"threshold": threshold,
			}

			# Calculate metrics
			classif_info = classif_info_from_exp_name(task)
			if classif_info['type_classif'] == 'several_concepts':
				for i_concept, concept in enumerate(classif_info['classif_classes']):
					y_true = combination_df[f'label_int_{concept}']
					y_pred = combination_df[f'predictions_{concept}']
					y_scores = combination_df[[f'test_proba_{concept}']].to_numpy()
					dict_metrics = calculate_metrics(y_true, y_pred, y_scores)
					new_dict_metrics = {f"{k}_{concept}": v for k, v in dict_metrics.items()}
					res[str_combination].update(new_dict_metrics)

			else:
				y_true = combination_df['label_int']
				y_pred = combination_df['predictions']
				y_scores = combination_df[test_proba_columns].to_numpy()
				dict_metrics = calculate_metrics(y_true, y_pred, y_scores)
				res[str_combination].update(dict_metrics)

			for n, v in zip(discriminators, combination):
				res[str_combination][n] = v
			
			if with_concept_props:
				nb_total = len(combination_df)
				for concept in ALL_CONCEPTS:
					concept_present = combination_df['concepts'].apply(lambda clip_concepts: any(x in [concept] for x in clip_concepts))
					nb_concept_present = int(np.count_nonzero(concept_present))
					res[str_combination][f"prop {concept}"] = nb_concept_present/nb_total

			res[str_combination][f"concept_density"] = float(np.mean(combination_df['concepts'].apply(len)))

			# For significance analysis, we may want to consider various factors, so we include them
			res[str_combination][f"ce_loss"] = ce_loss
			res[str_combination][f"differential"] = differential
			res[str_combination][f"variant"] = variant
			res[str_combination][f"core_architecture"] = core_architecture

		# We build the DataFrame from the dictionary of results
	res_df = pd.DataFrame.from_dict(res, orient='index')
	# For pretty print (ie. discarding seed and folds from mean and std computations), we change their type to str
	res_df.astype({'fold': str, 'seed': str, 'exp_name': str, 'dataset_name': str, 'model': str, 'train_annotator': str, 'test_annotator': str,
							'representation_architecture': str, 'representation_loss':str, 'ce_loss': str, 'differential': str,
							'core_architecture': str, 'variant': str, 'modality': str, 'spec_str': str
							})
	if 'movie_title' in res_df.columns:
		res_df = res_df.astype({'movie_title': str})
	if 'label' in res_df.columns:
		res_df = res_df.astype({'label': str})
	res_df = res_df.reset_index(drop=True)

	# Global results
	df_global = res_df.mean(numeric_only=True)
	df_global = pd.DataFrame(df_global).transpose()
	# We still want to keep the str columns that were constants
	df_global["exp_name"] = task
	df_global["dataset_name"] = dataset_name
	df_global["model"] = model_name
	df_global["modality"] = modality
	df_global["spec_str"] = spec_str
	df_global["train_annotator"] = train_annotator
	df_global["test_annotator"] = test_annotator
	# As those are constants, we can use the last str_combination in memory to access other values of interest :
	df_global['representation_architecture'] = representation_architecture
	df_global['representation_loss'] = representation_loss
	df_global['ce_loss'] = ce_loss
	df_global['differential'] = differential
	df_global['core_architecture'] = core_architecture
	df_global['variant'] = variant

	result_df = res_df if detailed else df_global

	#print(f"\n\n#######################\n{model_name}")
	#print(result_df.round(3).to_string())

	return result_df


def get_task_result(experiment, model_name, test_annotator, task, per_movie, per_label, detailed=True, with_concept_props=False,
					variable_threshold=False, fold_dataset=False,
					inference_set='test',
					suffix='',
					only_one_fold=False):
	base_model_predictions = get_prediction_df(**experiment,
											   test_annotator=test_annotator,
									# OTHERS
									variable_threshold=variable_threshold,
									new_classif_info=classif_info_from_exp_name(task),
											   fold_dataset=fold_dataset,
											   inference_set=inference_set,
											   suffix=suffix,
											   only_one_fold=only_one_fold
									)

	return calculate_metrics_on_df(base_model_predictions, model_name,
								   modality=experiment['modality'],
								   spec_str=experiment['spec_str'],
								   dataset_name=experiment['dataset_name'],
								   representation_architecture=experiment['representation_architecture'],
								   representation_loss=experiment['representation_loss'],
								   type_representation=experiment["type_representation"],
								   task=task, seed=experiment['seed'], train_annotator=experiment['train_annotator'], test_annotator=test_annotator,
								   per_label=per_label, per_movie=per_movie, detailed=detailed, with_concept_props=with_concept_props,
								   fold_dataset=fold_dataset)


def plot_task_barplot_accuracy(experiment, model_name, task, test_annotator=None, inference_set='test', suffix='', rotation=90):
	test_annotator = experiment['train_annotator'] if test_annotator is None else test_annotator
	train_annotator = experiment['train_annotator']
	exp_name = experiment['exp_name']
	result_df = get_task_result(experiment=experiment,
								test_annotator=test_annotator,
							 model_name=model_name, task=task,
							 per_label=True, per_movie=True,
								inference_set=inference_set,
								suffix=suffix)
	#print(f"result_df.to_string() :\n{result_df.to_string()}")

	# Plot
	labels = [0, 1]
	# Will play the role of hue_order to guarantee consistent labeling
	movies = result_df.movie_title.value_counts().sort_index(ascending=True).index.to_list()
	# Similarly, for reproducibility of plots, this will be the fixed order of movies (actually the alphabetical order)
	fig, ax = plt.subplots(figsize=(10, 8))
	sns.barplot(y="accuracy", x="movie_title", order=movies, data=result_df, hue='label_int', hue_order=labels,
				ax=ax)  # N.B. : here the 'label' is the binary one
	ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
	ax.set_ylim(0., 1.)

	plt.tight_layout()
	plot_dir = f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{train_annotator}/test_{test_annotator}/barplot_accuracy_{task}_by_movie_title_label_{inference_set}{suffix}"
	if not os.path.exists(plot_dir):
		os.makedirs(plot_dir)
	plt.savefig(f"{plot_dir}/{model_name.split('/')[-1]}_barplot_accuracy_{inference_set}{suffix}.png")
	tikzplotlib.save(f"{plot_dir}/{model_name.split('/')[-1]}_barplot_accuracy_{inference_set}{suffix}.tex")
	plt.close()


def get_concepts_performances(experiment, model_name, original_task, test_annotator=None):
	test_annotator = experiment['train_annotator'] if test_annotator is None else test_annotator
	train_annotator = experiment['train_annotator']
	exp_name = experiment['exp_name']
	result_df = get_task_result(experiment=experiment,
								test_annotator=test_annotator,
								model_name=model_name,
								task=original_task,
								per_movie=True,
								per_label=False,
								with_concept_props=True)

	print(result_df.round(3).to_string())

	mean_concept_precisions = dict()

	for concept in ALL_CONCEPTS:
		# We should not consider accuracies, given the very rather low proportion of these concepts in the full dataset
		# (so that a trivial constant negative classifier would achieve high accuracy)
		result_concept_df = get_task_result(experiment=experiment,
											test_annotator=test_annotator,
											model_name=model_name,
											task=concept,
											per_movie=True,
											per_label=False,
											with_concept_props=False)[
			["movie_title", "f1_score", "accuracy", "precision", "recall"]].rename(
			columns={'f1_score': f'f1_score {concept}',
					 'accuracy': f'accuracy {concept}',
					 'precision': f'precision {concept}',
					 'recall': f'recall {concept}'})

		concept_df = result_df[["movie_title", "f1_score", "accuracy", "precision", "recall", "concept_density", f"prop {concept}"]]
		concept_df = concept_df.merge(result_concept_df, on='movie_title')

		mean_concept_precisions[concept] = round(float(np.mean(concept_df[f"precision {concept}"])),3)

		print(f"\n\n############ {concept}\n{concept_df.round(3).to_string()}")

		# We also store those results
		# We create the directory for these results (different directory from inference results, for clarity)
		sum_up_directory = (f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{train_annotator}/test_{test_annotator}/"
							f"concept_performance")
		if not os.path.exists(sum_up_directory):
			os.makedirs(sum_up_directory)
		# We save the sum-up tables
		with open(f'{sum_up_directory}/{original_task}_on_{concept}.txt', 'w') as file:
			file.write(concept_df.round(3).to_string())
		# Also in .csv files
		concept_df.round(3).to_csv(f'{sum_up_directory}/{original_task}_on_{concept}.csv', sep=';')

		for couple in [("f1_score", f"f1_score {concept}"), ("f1_score", f"prop {concept}"), ("accuracy", f"prop {concept}"),
					   (f"prop {concept}", f"recall {concept}"), (f"prop {concept}", f"precision {concept}")]:
			print(f"\nCorrelations {couple}")
			print(f"Pearson : {round(pearsonr(concept_df[couple[0]], concept_df[couple[1]])[0], 3)}")
			print(f"Spearman : {round(spearmanr(concept_df[couple[0]], concept_df[couple[1]])[0], 3)}")

		# We do the same without The Help (that may harm results)
		print(f"\nAnd now same metrics without The Help :")
		without_the_help = concept_df[concept_df["movie_title"] != "The Help"]
		print(len(without_the_help))
		for couple in [("f1_score", f"f1_score {concept}"), ("f1_score", f"prop {concept}"), ("accuracy", f"prop {concept}"),
					   (f"prop {concept}", f"recall {concept}"), (f"prop {concept}", f"precision {concept}")]:
			print(f"\nCorrelations {couple}")
			print(f"Pearson : {round(pearsonr(without_the_help[couple[0]], without_the_help[couple[1]])[0], 3)}")
			print(f"Spearman : {round(spearmanr(without_the_help[couple[0]], without_the_help[couple[1]])[0], 3)}")

		# The plots asked by Lucile
		plot_dir = f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{train_annotator}/test_{test_annotator}/perf_of_prop_concept/{model_name.split('/')[-1]}"
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)

		movies = concept_df.movie_title.value_counts().sort_index(ascending=True).index.to_list()
		ax = sns.relplot(data=concept_df, x=f"prop {concept}", y="accuracy",
						 hue="movie_title",
						 hue_order=movies,
						 s=20)
		ax.figure.suptitle(f"Pearson correlation (accuracy, prop {concept}) (without The Help) : {round(pearsonr(without_the_help['accuracy'], without_the_help[f'prop {concept}'])[0], 3)}")
		#plt.tight_layout()
		plt.savefig(f"{plot_dir}/accuracy_of_prop_{concept}.png", dpi=200)
		plt.close()
		ax = sns.relplot(data=concept_df, x=f"prop {concept}", y="f1_score",
						 hue="movie_title",
						 hue_order=movies,
						 s=20)
		ax.figure.suptitle(f"Pearson correleation (F1-score, prop {concept}) (without The Help) : {round(pearsonr(without_the_help['f1_score'], without_the_help[f'prop {concept}'])[0], 3)}")
		#plt.tight_layout()
		plt.savefig(f"{plot_dir}/f1_score_of_prop_{concept}.png", dpi=200)
		plt.close()

	mean_concept_precisions = {k: mean_concept_precisions[k] for k in sorted(mean_concept_precisions, key=mean_concept_precisions.get)}
	print(f"\nmean_concept_precisions :\n{mean_concept_precisions}")


def print_worst_confidences(experiment, movies, body_restricted = False, new_classif_info=None,
							restrict_to_class=None, test_annotator=None):
	test_annotator = experiment['train_annotator'] if test_annotator is None else test_annotator
	base_model_predictions = get_prediction_df(**experiment,
											   test_annotator=test_annotator,
									# OTHERS
									variable_threshold=False,
									new_classif_info=new_classif_info
									)

	important_concepts = ['Clothing', 'Body']
	base_model_predictions['target_concepts'] = base_model_predictions['concepts'].apply(lambda clip_concepts: any(x in important_concepts for x in clip_concepts))
	if body_restricted:
		base_model_predictions = base_model_predictions[base_model_predictions['target_concepts']] # we only keep the rows containing important concepts
	base_model_predictions = base_model_predictions.drop(columns = ['target_concepts', 'framerate', 'start_clip', 'end_clip', 
                                                                 'fold', 'comment', 'annotator', 'clip_index', 'movie_id',
                                                                 'predictions',
																	'test_proba',
																	])

	for movie in movies :
		print(f"\n\n######## {movie}")
		nb_extreme = 10
		if restrict_to_class is not None:
			metrics_df = base_model_predictions.query('movie_title == @movie and label_int == @restrict_to_class')
		else:
			metrics_df = base_model_predictions
		extreme_indices = np.argsort(metrics_df["confidence"])
		lowest_indices, largest_indices = extreme_indices[:nb_extreme], extreme_indices[-nb_extreme:]
		lowest_df = metrics_df.iloc[lowest_indices]
		largest_df = metrics_df.iloc[largest_indices[::-1]]
		#print(f"\nPositive with the least confidence :\n{lowest_df.to_string()}")
		print(f"\nExamples ordered by confidence (restricted to class {restrict_to_class}) :\n"
			  f"{metrics_df.iloc[extreme_indices].to_string()}")


def main_performances(all_experiments, test_annotators, saving_str='', average_across_seeds=True,
					  variable_threshold=False):
	# First, we make a summary of all models on their global results
	sum_up_df = None
	for experiment_name, experiment in all_experiments.items():
		print(f"\n\n### {experiment_name}")
		exp_name = experiment["exp_name"]
		classif_info = classif_info_from_exp_name(exp_name)
		for test_annotator in test_annotators:
			# General results :
			exp_df = get_task_result(experiment=experiment, test_annotator=test_annotator, model_name=experiment_name,
									 task=experiment['exp_name'], variable_threshold=variable_threshold,
									 per_movie=False, per_label=False, detailed=False)
			# For computing heterogeneity :
			exp_movie_df = get_task_result(experiment=experiment, test_annotator=test_annotator, model_name=experiment_name,
										   task=experiment['exp_name'], variable_threshold=variable_threshold,
										   per_movie=True, per_label=False, detailed=True)

			if classif_info['type_classif'] != 'several_concepts':
				# Enjoying the occasion to plot accuracy barplot :
				exp_movie_label_df = get_task_result(experiment=experiment, test_annotator=test_annotator,
				                                     model_name=experiment_name,
				                                     task=experiment['exp_name'], variable_threshold=variable_threshold,
				                                     per_movie=True, per_label=True, detailed=True)
				# Plot
				# Will play the role of hue_order to guarantee consistent labeling
				movies = exp_movie_label_df.movie_title.value_counts().sort_index(ascending=True).index.to_list()
				# Similarly, for reproducibility of plots, this will be the fixed order of movies (actually the alphabetical order)
				fig, ax = plt.subplots(figsize=(10, 8))
				sns.barplot(y="accuracy", x="movie_title", order=movies, data=exp_movie_label_df, hue='label_int', hue_order=[0, 1],
							ax=ax)  # N.B. : here the 'label' is the binary one
				ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
				ax.set_ylim(0., 1.)

				plt.tight_layout()
				plot_dir = f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{experiment['train_annotator']}/test_{test_annotator}/barplot_accuracy_by_movie_title_label"
				if not os.path.exists(plot_dir):
					os.makedirs(plot_dir)
				plt.savefig(f"{plot_dir}/{experiment_name.split('/')[-1]}.png")
				plt.close()

				# We do the same without the labels, to plot the measures used for heterogeneity
				# Similarly, for reproducibility of plots, this will be the fixed order of movies (actually the alphabetical order)
				fig, ax = plt.subplots(figsize=(10, 8))
				sns.barplot(y="accuracy", x="movie_title", order=movies, data=exp_movie_df,
							ax=ax)  # N.B. : here the 'label' is the binary one
				ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
				ax.set_ylim(0., 1.)
				plt.tight_layout()
				plot_dir = f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{experiment['train_annotator']}/test_{test_annotator}/barplot_accuracy_by_movie_title"
				if not os.path.exists(plot_dir):
					os.makedirs(plot_dir)
				plt.savefig(f"{plot_dir}/{experiment_name.split('/')[-1]}.png")
				plt.close()

				accuracies = exp_movie_df['accuracy']
				a = accuracies > accuracies.min()
				b = accuracies < accuracies.max()
				truncated_accuracies = accuracies[a & b]
				exp_df['std_acc_across_movies'] = np.std(accuracies)
				# exp_df['trunc_std_acc_across_movies'] = np.std(truncated_accuracies)

			exp_df['model'] = experiment_name.split('/')[-1]
			sum_up_df = exp_df if sum_up_df is None else pd.concat([sum_up_df, exp_df], ignore_index=True)

	if sum_up_df["exp_name"].nunique(dropna=False) == 1:
		exp_name = sum_up_df["exp_name"].iat[0]
	else:
		exp_name = "various_exps"

	if average_across_seeds:
		sum_up_df = sum_up_df.groupby(['model', 'train_annotator', 'test_annotator', 'exp_name', 'dataset_name']).mean(numeric_only=True).reset_index()
	# 'train_annotator', 'test_annotator', 'exp_name' should be constant in the df (or at least some of them) :
	# I include them so they are not removed by the numeric_only=True

	if classif_info['type_classif'] != 'several_concepts':
		sum_up_df = sum_up_df.sort_values(["f1_weighted", "accuracy"], axis=0, ascending=False)
		sum_up_df = sum_up_df.reset_index(drop=True)
		# We want to put the model column on front :
		cols = sum_up_df.columns.tolist()
		rearranged_cols = ["exp_name", "dataset_name", "model", "train_annotator", "test_annotator",
						   "f1_macro", "accuracy", #"f1_score", "f1_negative", "f1_weighted",
						   #"precision", "precision_negative", "recall", "recall_negative",
						   "negative_pr_auc", "positive_pr_auc", "auc_pr_macro", "roc_auc", "std_acc_across_movies",
						   "threshold"]
		# N.B. : this line is not robust to modifs
		sum_up_df = sum_up_df[rearranged_cols] # this way we reorder columns and we also perform dropping (of 'concept_density', 'nb_obs')

	print(f"\n\n\n############### GLOBAL RESULTS ############")
	print(sum_up_df.round(3).to_string())

	# We also store those results
	# We create the directory for these results (different directory from inference results, for clarity)
	sum_up_directory = f'{PROJECT_DIR}/two_stage_sum_up_results'
	if not os.path.exists(sum_up_directory):
		os.makedirs(sum_up_directory)
	# We save the sum-up tables
	with open(f'{sum_up_directory}/sum_up_{exp_name}{saving_str}.txt', 'w') as file:
		file.write(sum_up_df.round(3).to_string())
	# Also in .csv files
	sum_up_df.round(3).to_csv(f'{sum_up_directory}/sum_up_{exp_name}{saving_str}.csv', sep=';')

	if classif_info['type_classif'] != 'several_concepts':
		# We do the trade-off maps, separated by train_annotator for them to be comparable (and tested on it)
		for train_annotator in set(e['train_annotator'] for e in all_experiments.values()):
			sum_up_df_annotator = sum_up_df.query("train_annotator == @train_annotator and test_annotator == @test_annotator")

			# global plot (from which I can get the x and y min and max)
			ax = sns.relplot(data=sum_up_df_annotator, x='std_acc_across_movies', y="accuracy",
							 style="model",
							 hue="model",
							 # hue_order=models,
							 s=20)
			# plt.setp(ax._legend.get_texts(), fontsize='5') # changing the fontsize is not enough, we have to reduce the df
			plot_dir = f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{train_annotator}/test_{train_annotator}/tradeoff_map_accuracy_heterogeneity"
			if not os.path.exists(plot_dir):
				os.makedirs(plot_dir)
			plt.savefig(f"{plot_dir}/tradeoff_map_accuracy_heterogeneity_all_models.png", dpi=200)
			plt.close()

			# global plot on good models (from which I can get the x and y min and max)
			ax = sns.relplot(data=sum_up_df_annotator, x='std_acc_across_movies', y="accuracy",
							 style="model",
							 hue="model",
							 # hue_order=models,
							 s=20)
			# plt.setp(ax._legend.get_texts(), fontsize='5') # changing the fontsize is not enough, we have to reduce the df
			plot_dir = f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{train_annotator}/test_{train_annotator}/tradeoff_map_accuracy_heterogeneity"
			if not os.path.exists(plot_dir):
				os.makedirs(plot_dir)
			plt.savefig(f"{plot_dir}/tradeoff_map_accuracy_heterogeneity.png", dpi=200)
			plt.close()


def confidence_plot(all_experiments, test_annotators):
	for experiment_name, experiment in all_experiments.items():
		train_annotator = experiment['train_annotator']
		exp_name = experiment['exp_name']
		for test_annotator in test_annotators:
			pred_df = get_prediction_df(**experiment,
										test_annotator=test_annotator,
										# OTHERS
										variablethreshold=False,
										)

			# plot
			print(f"\n\n{experiment_name.split('/')[-1]}")
			print(pred_df.round(3).head().to_string())

			# Will play the role of hue_order to guarantee consistent labeling
			movies = pred_df.movie_title.value_counts().sort_index(ascending=True).index.to_list()
			# Similarly, for reproducibility of plots, this will be the fixed order of movies (actually the alphabetical order)
			fig, ax = plt.subplots(figsize=(10, 8))
			sns.boxplot(y="confidence", x="movie_title", order=movies, data=pred_df, hue='label_int', hue_order=[0, 1],
						ax=ax)  # N.B. : here the 'label' is the binary one
			ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

			plt.tight_layout()
			plot_dir = (f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{train_annotator}/test_{test_annotator}/"
						f"boxplot_confidence_by_movie_title_label")
			if not os.path.exists(plot_dir):
				os.makedirs(plot_dir)
			plt.savefig(f"{plot_dir}/{experiment_name.split('/')[-1]}.png")
			plt.close()
