from two_stage.two_stage_train import two_stage_train

from constants import *
from global_utils import classif_info_from_exp_name


original_seed = 42
nb_seeds = 5
random_seeds = list(range(original_seed, original_seed + nb_seeds))

if __name__ == "__main__":
	batch_size = 64
	for exp_name in ["EN_HNS"]:
		print(f"\n\n\n################ (multimodal) ViT model | batch_size {batch_size}")
		two_stage_train(modality="text+visual",
						gt_csv=f"{DATASETS_DIR}/mobygaze_dataframe_soft_text+visual.csv",
						random_seeds=random_seeds,
						train_annotators=['annotator_2'],
						test_annotators=['annotator_2'],
						num_folds=NUM_FOLDS,
						folds_info=FOLDS_INFO,
						pool="cls-pooling",
						spec_str="_DMLR",
						exp_name=exp_name,
						######################################
						# TO MODIFY
						gpu_index=1,
						# TYPE OF REPRESENTATION TRAINING
						representation_architecture="multimodal_vit", representation_loss="CE",
						# TYPE OF CLASSIFICATION TRAINING
						consider_classif=False, type_classif="mlp",
						# OTHERS
						balancing=True, save_model=True,
						batch_size=batch_size,
						use_trained_representation_models=True,
						use_trained_classification_models=True
						)
