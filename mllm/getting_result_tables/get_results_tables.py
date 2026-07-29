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
#import tikzplotlib

from constants import *
from global_utils import *

pd.options.mode.copy_on_write = True


def get_prediction_df(model_path, prompt_characteristics, shot_str, dataset_name, exp_name, test_annotator, seed, spec_str,
					  inference_set='test', # can be 'test', 'val'
					  modality="visual",
					  new_classif_info=None,
					  **kwargs
					  ):
	model_str = model_path.split('/')[-1]
	full_exp_name = f"{prompt_characteristics['exp_name']}_{prompt_characteristics['spec']}"

	total_df = None

	# Instead of taking folds = NUM_FOLDS, the most rigorous is to check which results files actually exist
	folds = []
	for fold_idx in NUM_FOLDS:
		if model_path == "random":
			csv_directory = f"{PROJECT_DIR}/inference_results/{full_exp_name}/{shot_str}/Qwen2.5-VL-7B-Instruct/{dataset_name}_{modality}"
			combined_csv_path = f"{csv_directory}/{inference_set}_results_seed_{seed}_{test_annotator}_{exp_name}_fold_{fold_idx}_.csv"
		else:
			csv_directory = f"{PROJECT_DIR}/inference_results/{full_exp_name}/{shot_str}/{model_str}/{dataset_name}_{modality}"
			combined_csv_path = f"{csv_directory}/{inference_set}_results_seed_{seed}_{test_annotator}_{exp_name}_fold_{fold_idx}_{spec_str}.csv"
		if os.path.exists(combined_csv_path):
			folds.append(fold_idx)

	for fold_idx in folds:
		# As we have run no actual "random" experiment we generate artificial random results here.
		# In order to load the correct test set, we load inference results for the standard experiment
		if model_path == "random":
			csv_directory = f"{PROJECT_DIR}/inference_results/{full_exp_name}/{shot_str}/Qwen2.5-VL-7B-Instruct/{dataset_name}_{modality}"
			combined_csv_path = f"{csv_directory}/{inference_set}_results_seed_{seed}_{test_annotator}_{exp_name}_fold_{fold_idx}_.csv"
		else:
			csv_directory = f"{PROJECT_DIR}/inference_results/{full_exp_name}/{shot_str}/{model_str}/{dataset_name}_{modality}"
			combined_csv_path = f"{csv_directory}/{inference_set}_results_seed_{seed}_{test_annotator}_{exp_name}_fold_{fold_idx}_{spec_str}.csv"

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

		# We aggregate the folds dfs
		total_df = df if total_df is None else pd.concat([total_df, df], ignore_index=True)

	if 'subtitle' in total_df.columns:
		total_df = total_df.drop(columns=['subtitle'])

	return total_df


def calculate_metrics(y_true, y_pred, y_original_pred=None):
	num_output_neurons = max(y_true)
	# we assume that for multiclass classification, y_true contains at least one element of class 2

	if y_original_pred is not None:
		count_none = y_original_pred.isna().sum()
		prop_none = count_none / len(y_original_pred)
		prop_none = round(prop_none, 3)
	else:
		prop_none = None

	if num_output_neurons <= 1:

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

		binary_classification_metrics = {"accuracy": round(accuracy, 3),
				"f1_score": round(f1, 3),
				"f1_negative": round(f1_negative, 3),
				"f1_macro": round(f1_macro, 3),
				"f1_weighted": round(f1_w, 3),
				"precision": round(precision, 3),
				"precision_negative": round(precision_negative, 3),
				"recall": round(recall, 3),
				"recall_negative": round(recall_negative, 3),
				"true_positive_prop": round(true_positive_prop, 3),
				"pred_positive_prop": round(pred_positive_prop, 3),
				"prop_none": prop_none
										 }
		return binary_classification_metrics

	else:
		accuracy = accuracy_score(y_true, y_pred)
		multiclass_classification_metrics = {"accuracy": round(accuracy, 3), "prop_none": prop_none}
		return multiclass_classification_metrics


def calculate_metrics_on_df(df, model_path, prompt_characteristics, shot_str, task, dataset_name, seed, modality, spec_str, test_annotator=None,
							per_label=False, per_movie=False, detailed=True,
							with_concept_props=False): # detailed=False averages all lines

	model_str = model_path.split('/')[-1]
	full_exp_name = f"{prompt_characteristics['exp_name']}_{prompt_characteristics['spec']}"

	if test_annotator == "UNION":
		pass
	elif test_annotator is not None:
		df = df[df['annotator'] == test_annotator]

	res = dict()
	folds = df["fold"].unique().tolist()

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
			y_true = combination_df['label_int']
			y_pred = combination_df['test_pred']
			y_original_pred = combination_df['original_pred']

			# Calculate metrics
			dict_metrics = calculate_metrics(y_true, y_pred, y_original_pred)

			res[str_combination] = {
				"exp_name": task,
				"full_exp_name": full_exp_name,
				"spec_str": spec_str, # may be useful to make group-by's for experiments which differ only in spec_str
				"dataset_name": dataset_name,
				"model": model_str,
				"shot_str": shot_str,
				"modality":modality,
				"test_annotator": test_annotator,
				"seed": seed,
				"nb_obs": len(combination_df),
			}
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

	# We build the DataFrame from the dictionary of results
	res_df = pd.DataFrame.from_dict(res, orient='index')
	# For pretty print (ie. discarding seed and folds from mean and std computations), we change their type to str
	res_df.astype({'fold': str, 'seed': str, 'exp_name': str, 'full_exp_name': str, 'dataset_name': str, 'model': str, 'test_annotator': str,
					'modality': str, 'spec_str': str, 'shot_str': str})
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
	df_global["full_exp_name"] = full_exp_name
	df_global["dataset_name"] = dataset_name
	df_global["model"] = model_str
	df_global["shot_str"] = shot_str
	df_global["modality"] = modality
	df_global["spec_str"] = spec_str
	df_global["test_annotator"] = test_annotator

	result_df = res_df if detailed else df_global

	#print(f"\n\n#######################\n{model_name}")
	#print(result_df.round(3).to_string())

	return result_df


def get_task_result(experiment, model_path, test_annotator, task, per_movie, per_label, detailed=True,
					with_concept_props=False,
					inference_set='test',):
	base_model_predictions = get_prediction_df(**experiment,
											   test_annotator=test_annotator,
											   inference_set=inference_set,
											   new_classif_info=classif_info_from_exp_name(task)
									)

	return calculate_metrics_on_df(base_model_predictions, model_path,
								   prompt_characteristics=experiment['prompt_characteristics'],
								   shot_str=experiment['shot_str'],
	                               task=task, dataset_name=experiment['dataset_name'], seed=experiment['seed'],
	                               modality=experiment['modality'], spec_str=experiment['spec_str'],
								   test_annotator=test_annotator,
								   per_label=per_label, per_movie=per_movie, detailed=detailed,
								   with_concept_props=with_concept_props,
								   )


def plot_task_barplot_accuracy(experiment, model_path, task,
							   test_annotator=None, inference_set='test', suffix='', rotation=90):
	exp_name = experiment['exp_name']
	shot_str = experiment["shot_str"]
	model_str = model_path.split('/')[-1]
	dataset_name = experiment["dataset_name"]
	modality = experiment["modality"]
	result_df = get_task_result(experiment=experiment, model_path=model_path,
								test_annotator=test_annotator, task=task,
								per_movie=True, per_label=True,
								inference_set=inference_set,)
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
	plot_dir = f"{PROJECT_DIR}/plots/{shot_str}/{model_str}/{dataset_name}_{modality}/{exp_name}/test_{test_annotator}/barplot_accuracy_{task}_by_movie_title_label"
	if not os.path.exists(plot_dir):
		os.makedirs(plot_dir)
	plt.savefig(f"{plot_dir}/{model_str}_barplot_accuracy_{inference_set}.png")
	#tikzplotlib.save(f"{plot_dir}/{model_name.split('/')[-1]}_barplot_accuracy_{inference_set}.tex")
	plt.close()


def main_performances(all_experiments, test_annotators, saving_str='', average_across_seeds=True,
					  new_task=None):
	# First, we make a summary of all models on their global results
	sum_up_df = None
	for experiment_name, experiment in all_experiments.items():
		print(f"\n\n### {experiment_name}")
		exp_name = experiment["exp_name"]
		shot_str = experiment["shot_str"]
		model_path = experiment["model_path"]
		model_str = model_path.split('/')[-1]
		dataset_name = experiment["dataset_name"]
		modality = experiment["modality"]

		for test_annotator in test_annotators:
			task = experiment['exp_name'] if new_task is None else new_task
			# General results :
			exp_df = get_task_result(experiment=experiment, test_annotator=test_annotator, model_path=model_path, task=task, per_movie=False, per_label=False, detailed=False)
			# For computing heterogeneity :
			exp_movie_df = get_task_result(experiment=experiment, test_annotator=test_annotator, model_path=model_path, task=task, per_movie=True, per_label=False, detailed=True)
			# Enjoying the occasion to plot accuracy barplot :
			exp_movie_label_df = get_task_result(experiment=experiment, test_annotator=test_annotator, model_path=model_path, task=task, per_movie=True, per_label=True, detailed=True)

			# print(f"\n\n###{experiment_name.split('/')[-1]}\n{exp_movie_df.round(3).to_string()}")
			# print(f"\n\n###{experiment_name.split('/')[-1]}\n{exp_movie_label_df.round(3).to_string()}")

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
			plot_dir = f"{PROJECT_DIR}/plots/{shot_str}/{model_str}/{dataset_name}_{modality}/{task}/test_{test_annotator}/barplot_accuracy_by_movie_title_label"
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
			plot_dir = f"{PROJECT_DIR}/plots/{shot_str}/{model_str}/{dataset_name}_{modality}/{task}/test_{test_annotator}/barplot_accuracy_by_movie_title"
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

			# From exp_movie_df, we can also construct the "plot" of accuracy per movie

			exp_df['model'] = experiment_name.split('/')[-1]
			sum_up_df = exp_df if sum_up_df is None else pd.concat([sum_up_df, exp_df], ignore_index=True)

	exp_name = sum_up_df["exp_name"].iat[0] if new_task is None else new_task

	if average_across_seeds:
		sum_up_df = sum_up_df.groupby(['model', 'test_annotator', 'exp_name', 'dataset_name', 'full_exp_name', 'spec_str']).mean(numeric_only=True).reset_index()
	# 'train_annotator', 'test_annotator', 'exp_name' should be constant in the df (or at least some of them) :
	# I include them so they are not removed by the numeric_only=True

	sum_up_df = sum_up_df.sort_values(["f1_macro"], axis=0, ascending=False)
	sum_up_df = sum_up_df.reset_index(drop=True)
	# We want to put the model column on front :
	cols = sum_up_df.columns.tolist()
	#rearranged_cols = []
	# N.B. : this line is not robust to modifs
	#sum_up_df = sum_up_df[rearranged_cols] # this way we reorder columns and we also perform dropping (of 'concept_density', 'nb_obs')
	print(f"\n\n\n############### GLOBAL RESULTS ############")
	print(sum_up_df.round(3).to_string())

	# We also store those results
	# We create the directory for these results (different directory from inference results, for clarity)
	sum_up_directory = f'{PROJECT_DIR}/result_tables'
	if not os.path.exists(sum_up_directory):
		os.makedirs(sum_up_directory)
	# We save the sum-up tables
	with open(f'{sum_up_directory}/sum_up_{exp_name}{saving_str}.txt', 'w') as file:
		file.write(sum_up_df.round(3).to_string())
	# Also in .csv files
	sum_up_df.round(3).to_csv(f'{sum_up_directory}/sum_up_{exp_name}{saving_str}.csv', sep=';')
