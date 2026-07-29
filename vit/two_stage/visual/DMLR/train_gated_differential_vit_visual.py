from two_stage.two_stage_train import two_stage_train

from constants import *
from global_utils import classif_info_from_exp_name


original_seed = 42
nb_seeds = 5
random_seeds = list(range(original_seed, original_seed + nb_seeds))

if __name__ == "__main__":
	nb_movies_per_batch = 2
	nb_clips_per_movie = 32
	for exp_name in ["ENHN_S", "EN_S", "EN_HN_S"] + VISUAL_CONCEPTS: # ["EN_HNS", "ENHN_S", "EN_S", "EN_HN_S"] + VISUAL_CONCEPTS:
		print(f"\n\n\n################ gated differential ViT model | batch_size {nb_movies_per_batch} x {nb_clips_per_movie}")
		two_stage_train(modality="visual",
						gt_csv=f"{DATASETS_DIR}/mobygaze_dataframe_soft_visual.csv",
						random_seeds=random_seeds,
						train_annotators=['annotator_2', 'annotator_1'], #TODO: ['annotator_2', 'annotator_1']
						test_annotators=['annotator_2', 'annotator_1', 'UNION'], #TODO: ['annotator_2', 'annotator_1', 'UNION']
						num_folds=NUM_FOLDS,
						folds_info=FOLDS_INFO,
						pool="cls-pooling",
						spec_str="_DMLR",
						exp_name=exp_name,
						classif_info=classif_info_from_exp_name(exp_name),
						######################################
						# TO MODIFY
						gpu_index=2,
						# TYPE OF REPRESENTATION TRAINING
						representation_architecture="gated_differential_vit", representation_loss="CE",
						# DIFFERENTIAL TRAINING
						nb_movies_per_batch=nb_movies_per_batch, nb_clips_per_movie=nb_clips_per_movie,
						# TYPE OF CLASSIFICATION TRAINING
						consider_classif=False, type_classif="mlp",
						# OTHERS
						balancing=True, save_model=True,
						use_trained_representation_models=True,
						use_trained_classification_models=True
						)
