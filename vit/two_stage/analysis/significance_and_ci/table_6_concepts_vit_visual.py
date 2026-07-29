from two_stage.analysis.significance_and_ci.significance_analysis import *


if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	# First, we have to keep track of all the experiments we've done
	all_experiments = dict()  # will be a dictionary of experiments dictionaries
	# N.B. : for the sake of simplicity, an experiment dictionary may contain useless arguments
	# (for instance contrastive arguments for non-contrastive models) but that shouldn't have any consequence

	core_architecture = "vit"
	addon = ''
	representation_architecture = f"{addon}{core_architecture}"
	GLOBAL_TRAIN_ANNOTATOR = 'annotator_2'
	modality = 'visual'
	dataset_name = f'mobygaze_dataframe_soft_{modality}'

	original_seed = 42
	nb_seeds = 5
	random_seeds = list(range(original_seed, original_seed + nb_seeds))

	for exp_name in VISUAL_CONCEPTS:

		for seed in random_seeds:

			for train_annotator in [GLOBAL_TRAIN_ANNOTATOR]:
				exp_str = f'{exp_name}/{dataset_name}/{train_annotator}/{seed}'

				representation_loss = 'CE'
				aposteriori_backbone = {'representation_architecture': core_architecture, 'representation_loss': "CE",
						'global_contrastive': None, 'contrastive_type': None, 'weighted_contrastive': None,
						'temperature': None,
						'nb_movies_per_batch': None, 'nb_clips_per_movie': None,
						'nb_segment': None, 'length_segment': None,
						'target_model': None,
						'batch_size': 128, 'train_annotator_backbone': None, 'representation_backbone': None}

				nb_movies_per_batch = 2
				nb_clips_per_movie = 32
				batch_size = 128
				exp_dict = {'exp_name': exp_name, 'train_annotator': train_annotator,
							'seed': seed,
							# TYPE OF REPRESENTATION TRAINING
							'representation_architecture': representation_architecture,
							'representation_loss': representation_loss,
							'batch_size': batch_size,
							'full_model': True,
							# CONTRASTIVE ARGUMENTS
							'global_contrastive': None, 'contrastive_type': None,
							'weighted_contrastive': None, 'temperature': None,
							# TYPE OF CLASSIFICATION TRAINING
							'type_classif': None, 'batch_size_class': None,
							# DIFFERENTIAL TRAINING
							'nb_movies_per_batch': nb_movies_per_batch,
							'nb_clips_per_movie': nb_clips_per_movie,
							# MIL
							'nb_segment': None, 'length_segment': None,
							# TARGET EMBEDDING
							'target_model': None,
							# BACKBONE
							'train_annotator_backbone': None,
							'representation_backbone': aposteriori_backbone,
							# FOR DETAILED DIRECTORY
							'modality': modality, 'pool': 'cls-pooling', 'spec_str': '_DMLR',
							'dataset_name': dataset_name,
							}
				repr_str, type_representation = get_repr_strs(**exp_dict)
				exp_dict["type_representation"] = type_representation
				model_name = f'{repr_str}_full_model'
				all_experiments[f'{exp_str}/{model_name}'] = exp_dict

	# Now all_experiments contains all experiments done so far
	str_all_experiments = '\n'.join(list(all_experiments.keys()))
	print(f"We consider {len(all_experiments)} experiments (each containing 5 observations, for the 5 folds) :"
		  f"\n{str_all_experiments}\n")

	metrics = ["positive_pr_auc", "negative_pr_auc", "auc_pr_macro", 'std_acc_across_movies', "accuracy"]
	# ['accuracy', 'f1_score', 'f1_negative', 'f1_macro', 'f1_weighted', 'precision', 'precision_negative',
	# 'recall', 'recall_negative', 'roc_auc', 'negative_pr_auc', 'positive_pr_auc', 'auc_pr_macro',
	# 'std_acc_across_movies'] are all possible metrics
	standard_ci(all_experiments, spec_str=f'{representation_architecture}_{GLOBAL_TRAIN_ANNOTATOR}_{modality}_concepts',
				metrics=metrics,
				groupby_variable="exp_name", alpha=0.05, results_folder='DMLR')
