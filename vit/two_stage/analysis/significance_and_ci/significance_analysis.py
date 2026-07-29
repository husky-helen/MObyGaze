import pandas as pd
import warnings
import itertools
from copy import deepcopy
from math import sqrt

from two_stage.two_stage_train import get_repr_strs

pd.options.mode.copy_on_write = True
from constants import *
from two_stage.analysis.analyze_two_stage_disaggregated_results import get_task_result
from two_stage.analysis.significance_and_ci.statistical_testing import *
from global_utils import save_df_to_csv


# General function to make CI's (where the population for one model corresponds to its fold, seed)
# The group-by variable enables to compare other things than models :
# for instance, we can compare how a single model performs on various tasks (cf. Table 2 of DMLR paper)
def standard_ci(all_experiments, spec_str, metrics, groupby_variable, results_folder, alpha=0.05, variable_threshold=False,
				verbose=False, fold_dataset=False, inference_set='test', suffix='',
				return_df=False):
	# First, we make a summary of all models on their global results
	sum_up_df = None
	for experiment_name, experiment in all_experiments.items():
		test_annotator = experiment["train_annotator"]
		# General results :
		exp_df = get_task_result(experiment=experiment, test_annotator=test_annotator, model_name=experiment_name,
								 task=experiment['exp_name'], per_movie=False, per_label=False, detailed=True,
								 variable_threshold=variable_threshold, fold_dataset=fold_dataset, inference_set=inference_set, suffix=suffix)
		# For computing heterogeneity :
		exp_movie_df = get_task_result(experiment=experiment, test_annotator=test_annotator, model_name=experiment_name,
									   task=experiment['exp_name'], per_movie=True, per_label=False, detailed=True,
									   variable_threshold=variable_threshold, fold_dataset=fold_dataset, inference_set=inference_set, suffix=suffix)

		# N.B. : this time, for our purpose, we compute the heterogeneity at the level of the fold
		# In particular, the obtained values will be comparable with each other,
		# but they may not be comparable with previously obtained values (in the main_performances function)
		df_heterogeneity = exp_movie_df[['accuracy', 'fold']].groupby(['fold']).std(
			numeric_only=True).reset_index().rename(columns={'accuracy': 'std_acc_across_movies'})
		exp_df = pd.merge(exp_df, df_heterogeneity, on='fold')

		exp_df['model'] = experiment_name.split('/')[-1]
		sum_up_df = exp_df if sum_up_df is None else pd.concat([sum_up_df, exp_df], ignore_index=True)

	sum_up_df = sum_up_df.reset_index(drop=True)
	if verbose:
		print(f"sum_up_df :\n{sum_up_df.to_string()}\n")

	print(f"\n\n\nStandard confidence intervals") # we can do it for all metrics this time

	grouped_sum_up_df = sum_up_df.groupby([groupby_variable])

	def ci(input_df):
		data = {groupby_variable: input_df[groupby_variable].iat[0]}
		n = len(input_df)
		ppf = t.ppf(1 - alpha / 2, df=n-1)
		for metric in metrics:
			mean = input_df[metric].mean()
			std = input_df[metric].std() # the default is indeed the corrected sample std with n-1 df
			ci_width = sqrt(std/n) * ppf
			data[f"CI {metric}"] = f"{round(mean, 3)} +/- {round(ci_width, 3)}"

		return pd.DataFrame(data, index=[0])

	res_df = grouped_sum_up_df.apply(ci).reset_index(drop=True)

	print(res_df.to_string())
	csv_directory = f'{PROJECT_DIR}/two_stage_sum_up_results/{results_folder}'
	csv_name = f'{spec_str}_standard_CI_grouped_by_{groupby_variable}'
	save_df_to_csv(res_df, csv_directory, csv_name)

	if return_df:
		return res_df


# With this function, what we compare are models
def significance_analysis(all_experiments, spec_str, metrics, results_folder='', alpha=0.05, variable_threshold=False,
						  verbose=False, fold_dataset=False, factor='model',
						  inference_set='test', suffix='',
						  return_df=False):
	# First, we make a summary of all models on their global results
	sum_up_df = None
	for experiment_name, experiment in all_experiments.items():
		if "test_annotator" in experiment.keys():
			test_annotator = experiment["test_annotator"]
			del experiment["test_annotator"]
		else:
			test_annotator = experiment["train_annotator"]

		# General results :
		exp_df = get_task_result(experiment=experiment, test_annotator=test_annotator, model_name=experiment_name,
								 task=experiment['exp_name'], per_movie=False, per_label=False, detailed=True,
								 variable_threshold=variable_threshold, fold_dataset=fold_dataset, inference_set=inference_set, suffix=suffix)
		# For computing heterogeneity :
		exp_movie_df = get_task_result(experiment=experiment, test_annotator=test_annotator, model_name=experiment_name,
									   task=experiment['exp_name'], per_movie=True, per_label=False, detailed=True,
									   variable_threshold=variable_threshold, fold_dataset=fold_dataset, inference_set=inference_set, suffix=suffix)

		# N.B. : this time, for our purpose, we compute the heterogeneity at the level of the fold
		# In particular, the obtained values will be comparable with each other,
		# but they may not be comparable with previously obtained values (in the main_performances function)
		df_heterogeneity = exp_movie_df[['accuracy', 'fold']].groupby(['fold']).std(
			numeric_only=True).reset_index().rename(columns={'accuracy': 'std_acc_across_movies'})
		exp_df = pd.merge(exp_df, df_heterogeneity, on='fold')

		exp_df['model'] = experiment_name.split('/')[-1]
		sum_up_df = exp_df if sum_up_df is None else pd.concat([sum_up_df, exp_df], ignore_index=True)

	exp_name = sum_up_df["exp_name"].iat[0]
	sum_up_df = sum_up_df.reset_index(drop=True)

	if verbose:
		print(f"sum_up_df :\n{sum_up_df.to_string()}\n")

	# We create the feature fold * seed (as repeated measure takes a single column as 'subject')
	features_to_merge = ['fold', 'seed']
	merged_features_name = " * ".join(features_to_merge)
	sum_up_df[merged_features_name] = sum_up_df[features_to_merge].apply(lambda x: ','.join(x.astype(str)), axis=1)
	# We create the feature model * seed (as repeated measure takes a single column as 'subject')
	features_to_merge = ['model', 'seed']
	merged_features_name = " * ".join(features_to_merge)
	sum_up_df[merged_features_name] = sum_up_df[features_to_merge].apply(lambda x: ','.join(x.astype(str)), axis=1)
	# We create the feature fold * seed * variant (as repeated measure takes a single column as 'subject')
	features_to_merge = ['fold * seed', 'variant']
	merged_features_name = " * ".join(features_to_merge)
	sum_up_df[merged_features_name] = sum_up_df[features_to_merge].apply(lambda x: ','.join(x.astype(str)), axis=1)
	# We create the feature fold * seed * core_architecture (as repeated measure takes a single column as 'subject')
	features_to_merge = ['fold * seed', 'core_architecture']
	merged_features_name = " * ".join(features_to_merge)
	sum_up_df[merged_features_name] = sum_up_df[features_to_merge].apply(lambda x: ','.join(x.astype(str)), axis=1)
	# We create the feature model * modality (in case several modalities using the same model are present)
	features_to_merge = ['model', 'modality']
	merged_features_name = " * ".join(features_to_merge)
	sum_up_df[merged_features_name] = sum_up_df[features_to_merge].apply(lambda x: ','.join(x.astype(str)), axis=1)

	print(f"\nsum_up_df :\n{sum_up_df.to_string()}")
	# We have all experiments : from there we can construct the tables of interest

	print(f"\n\n\nANOVA-based confidence intervals")

	repeated_measure_subject = 'fold * seed'
	#N.B. : Unlike factor, which can be 'model' or 'model * modality', for now the subject cannot be chosen
	normalized_df = deepcopy(sum_up_df)

	dmlr_df = None

	for target_variable in metrics:
		print(f"\n\n## Significance of {factor} on {target_variable.upper()}")
		# We don't need it to compute ms_error,
		# but to check normality and homoscedasticity we still need to construct the normalized data
		mean_target = np.mean(normalized_df[target_variable])
		for s in sum_up_df[repeated_measure_subject].unique():
			mean_subject = np.mean(sum_up_df[sum_up_df[repeated_measure_subject] == s][target_variable])
			normalized_df.loc[normalized_df[repeated_measure_subject] == s, target_variable] += (mean_target - mean_subject)

		# We plot the distributions, to see visually why the distributions are not considered normal, if they are not
		#groups = sum_up_df[factor].value_counts().sort_index(ascending=True).index.to_list()
		#for df, name_df in zip([sum_up_df, normalized_df], ['original_data', 'normalized_data']):
			#	sns.histplot(x=target_variable, data=df, hue=factor, hue_order=groups, multiple="layer") # hue=factor, hue_order=groups, multiple="layer"
			#	# Saving
			#	plot_dir = (f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{experiment['train_annotator']}/test_{experiment['train_annotator']}/"
			#				f"perf_distribution_across_groups")
			#	if not os.path.exists(plot_dir):
			#		os.makedirs(plot_dir)
			#	plt.savefig(f"{plot_dir}/{target_variable}_across_{factor}_{name_df}.png",
			#			dpi=200)
			#	plt.close()

		# We do the actual testing of normality and homoscedasticity
		print(f"\nNormality and homoscedasticity tests :")
		# a) on original data
		homoscedasticity_checked = pg.homoscedasticity(data=sum_up_df, dv=target_variable, group=factor, alpha=alpha).at['levene', 'equal_var']
		normality_checked = all_groups_normal(sum_up_df, factor_str=factor, target_variable=target_variable, alpha=alpha)
		print(f"On original data :\nhomoscedasticity_checked : {homoscedasticity_checked}\nnormality_checked : {normality_checked}")
		# b) on normalized data
		homoscedasticity_checkedn = pg.homoscedasticity(data=normalized_df, dv=target_variable, group=factor, alpha=alpha).at['levene', 'equal_var']
		normality_checkedn = all_groups_normal(normalized_df, factor_str=factor, target_variable=target_variable,alpha=alpha)
		print(f"On normalized data :\nhomoscedasticity_checked : {homoscedasticity_checkedn}\nnormality_checked : {normality_checkedn}")

		res_rm_anova = pingouin.rm_anova(data=sum_up_df, dv=target_variable, within=factor, subject=repeated_measure_subject, detailed=True)
		res_anova = pingouin.anova(data=sum_up_df, dv=target_variable, between=factor, detailed=True)
		res_rm_anova.set_index('Source', inplace=True)
		res_anova.set_index('Source', inplace=True)
		print(f"\nres_rm_anova :\n{res_rm_anova}")
		print(f"\nres_anova (out of curiosity, to see what we gain by removing inter-subject variance) :\n{res_anova}")
		ms_error = res_rm_anova.at["Error", "MS"]
		ms_within = res_anova.at["Within", "MS"]
		df_error = res_rm_anova.at["Error", "DF"]
		n = len(sum_up_df[repeated_measure_subject].unique())
		ppf = t.ppf(1 - alpha/2, df=df_error)
		ci_width = np.sqrt(ms_error / (2*n)) * ppf
		print(f"\nms_error = estimation de sigma'^2 : {round(ms_error, 5)}")
		print(f"ci_width : {round(ci_width, 4)}")

		# The df of the mean results for each group
		df_mean = sum_up_df[[factor, target_variable]].groupby([factor]).mean(numeric_only=True).reset_index()

		# We still perform pairwise comparison in order to print Cohen's d
		useful_df_mean = df_mean.set_index(factor)
		models_to_test = sum_up_df[factor].unique()
		couples_to_test = list(itertools.combinations(models_to_test, 2))
		for x_name, y_name in couples_to_test:
			print(f"\n{x_name} VS {y_name} :")
			diff = useful_df_mean.at[x_name, target_variable] - useful_df_mean.at[y_name, target_variable]
			cohens_d = diff / (np.sqrt(ms_error))
			print(f"difference : {round(diff, 3)}")
			print(f"Cohen's d (on normalized data) : {round(cohens_d, 3)}")

		# We add the mean results, with their CI, to the DMLR table
		df_mean[target_variable] = df_mean[target_variable].apply(lambda x: round(x, 3))
		df_mean = df_mean.astype({target_variable: str})
		# We also print the confidence interval
		df_mean[target_variable] = df_mean[target_variable].apply(lambda x: x + f" +/- {round(ci_width, 3)}")
		# We add this target variable to the global DMLR table
		dmlr_df = df_mean if dmlr_df is None else pd.merge(dmlr_df, df_mean, how='inner', on=factor)

	print(f"\n\nDMLR table:\n{dmlr_df.to_string()}")
	csv_directory = f'{PROJECT_DIR}/two_stage_sum_up_results/{results_folder}'
	csv_name = f'{spec_str}_ANOVA_CI'
	save_df_to_csv(dmlr_df, csv_directory, csv_name)

	print(f"\n\n\nStandard confidence intervals") # we can do it for all metrics this time

	# Now train_test_sum_up_table is complete, we can compute the differences between diff and standard for each fold and seed
	grouped_sum_up_df = sum_up_df.groupby([factor])

	def ci(input_df):
		data = {factor: input_df[factor].iat[0]}
		n = len(input_df)
		ppf = t.ppf(1 - alpha / 2, df=n-1)
		for metric in metrics:
			mean = input_df[metric].mean()
			std = input_df[metric].std() # the default is indeed the corrected sample std with n-1 df
			ci_width = sqrt(std/n) * ppf
			data[f"CI {metric}"] = f"{round(mean, 3)} +/- {round(ci_width, 3)}"

		return pd.DataFrame(data, index=[0])

	res_df = grouped_sum_up_df.apply(ci).reset_index(drop=True)

	print(res_df.to_string())
	csv_directory = f'{PROJECT_DIR}/two_stage_sum_up_results/{results_folder}'
	csv_name = f'{spec_str}_standard_CI'
	save_df_to_csv(res_df, csv_directory, csv_name)

	if return_df:
		return res_df
