import warnings

from global_utils import save_df_to_csv, get_shot_str
from statistical_testing.significance_analysis import significance_analysis

if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	# First, we have to keep track of all the experiments we've done
	all_experiments = dict()  # will be a dictionary of experiments dictionaries
	# N.B. : for the sake of simplicity, an experiment dictionary may contain useless arguments
	# (for instance contrastive arguments for non-contrastive models) but that shouldn't have any consequence
	# N.B. : Be careful to let the actual model_name at the end of the all_experiments keys
	ANNOTATOR = 'annotator_2'
	prompt_characteristics = {'exp_name': 'EN_HNS', 'spec': 'final'}
	# FEW-SHOT ARGUMENTS
	# pooling_type = 'expert_annotated'  # 'base', 'expert_annotated', 'preliminary_few_shot'
	# selection_type = 'balanced_similarity'  # similarity
	zero_shot = True
	preliminary_res_name = None  # f"mobygaze_dataframe_annotator_2_num1_demo_pool_for_EN_HNS"
	few_shot_annotator = ANNOTATOR
	total_expert_budget = None  # 50
	expert_spec_str = ''
	pooling_type = 'base'

	selection_type = None
	few_shot_num = None
	few_shot_order = None

	model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

	original_seed = 42
	nb_seeds = 5
	random_seeds = list(range(original_seed, original_seed + nb_seeds))

	for seed in random_seeds:

		for modality in ["visual", "text", "text+visual"]:
			shot_str = get_shot_str(zero_shot=zero_shot, pooling_type=pooling_type, few_shot_annotator=few_shot_annotator,
									selection_type=selection_type, total_expert_budget=total_expert_budget,
									expert_spec_str=expert_spec_str, few_shot_num=few_shot_num,
									few_shot_order=few_shot_order,
									preliminary_res_name=preliminary_res_name)

			first_experiment = {'model_path': model_name,
								'prompt_characteristics': prompt_characteristics,
								'shot_str': shot_str,
								'exp_name': 'EN_HNS',
								'dataset_name': 'mobygaze_dataframe_soft_visual_corrected',
								'modality': modality,
								'seed': 42,
								'spec_str': "_multimodal_exp",
								'test_annotator': ANNOTATOR,}

			# exp_df = get_task_result(experiment=first_experiment, test_annotator=ANNOTATOR, model_path=MODEL,
			#                         task=first_experiment['exp_name'], per_movie=False, per_label=False, detailed=False)
			# print(exp_df.to_string())

			all_experiments[f"{seed}/{modality}/{model_name}"] = first_experiment

	# Now all_experiments contains all experiments done so far
	str_all_experiments = '\n'.join(list(all_experiments.keys()))
	print(f"We consider {len(all_experiments)} experiments (each containing 5 observations, for the 5 folds) :"
		  f"\n{str_all_experiments}\n")

	metrics = ['accuracy', 'f1_score', 'f1_negative', 'f1_macro', 'f1_weighted', 'precision', 'precision_negative',
			   'recall', 'recall_negative',
			   "true_positive_prop", "pred_positive_prop", "prop_none", 'std_acc_across_movies']
	# ['accuracy', 'f1_score', 'f1_negative', 'f1_macro', 'f1_weighted', 'precision', 'precision_negative',
	# 'recall', 'recall_negative', 'roc_auc', 'negative_pr_auc', 'positive_pr_auc', 'auc_pr_macro',
	# 'std_acc_across_movies'] are all possible metrics
	significance_analysis(all_experiments, alpha=0.05, metrics=metrics, factor='modality',
						  spec_str=f'compare_modalities', results_folder='DMLR',
						  verbose=False)
