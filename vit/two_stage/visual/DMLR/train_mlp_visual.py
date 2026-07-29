from two_stage.two_stage_train import two_stage_train

from constants import *
from global_utils import classif_info_from_exp_name


original_seed = 42
nb_seeds = 5
random_seeds = list(range(original_seed, original_seed + nb_seeds))

if __name__ == "__main__":
	batch_size = 128
	for exp_name in ["EN_HNS", "ENHN_S", "EN_S", "EN_HN_S"] + VISUAL_CONCEPTS:
		print(f"\n\n\n################ MLP model | batch_size {batch_size}")
		two_stage_train(modality="visual",
						gt_csv=f"{DATASETS_DIR}/mobygaze_dataframe_soft_visual_corrected.csv",
						random_seeds=random_seeds,
						train_annotators=['annotator_2'], # ['annotator_2', 'annotator_1']
						test_annotators=['annotator_2'], # ['annotator_2', 'annotator_1', 'UNION']
						num_folds=NUM_FOLDS,
						folds_info=FOLDS_INFO,
						pool="cls-pooling",
						spec_str="_new_features",
						exp_name=exp_name,
						######################################
						# TO MODIFY
						gpu_index=3,
						# TYPE OF REPRESENTATION TRAINING
						representation_architecture="mlp", representation_loss="CE",
						# TYPE OF CLASSIFICATION TRAINING
						consider_classif=False, type_classif="mlp",
						# OTHERS
						balancing=True, save_model=True,
						batch_size=batch_size,
						use_trained_representation_models=True,
						use_trained_classification_models=True
						)
