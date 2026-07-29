from constants import *

from inference.zero_shot_inference import zero_shot_inf


if __name__ == "__main__":
	csv_path = f"{DATA_DIR}/mobygaze_dataframe_soft_visual_corrected.csv"
	# WARNING : for multimodal experiments, you should use f"{DATA_DIR}/mobygaze_dataframe.csv"
	# so that the ground-truth labels, saved at inference time, correspond to the desired modality
	# The present code is erroneous, although corresponding results have been corrected subsequently using correct_labels.py
	# (we only let this csv_path out of consistency, since it is how results have been generated for the article)
	modality="text"
	test_annotator = 'annotator_2'
	num_folds = NUM_FOLDS
	gpu_index = 5 # N.B. : command without effect for now
	prompt_characteristics = {'exp_name': 'EN_HNS', 'spec': 'final'}

	original_seed = 42
	nb_seeds = 5
	random_seeds = list(range(original_seed, original_seed + nb_seeds))

	for seed in random_seeds:

		zero_shot_inf(  # TEST DATASET
			csv_path=csv_path,modality=modality, test_annotator=test_annotator, num_folds=num_folds, folds_info=FOLDS_INFO,
			# SAMPLING PARAMETERS
			num_frames="one_per_second", num_frames_max=50, max_pixels=64 * 28 * 28, align_factor=28,
			# MODEL PARAMETERS
			model_path= "Qwen/Qwen2.5-VL-7B-Instruct", max_model_len=8192, gpu_index=5,
			# INFERENCE PARAMETERS
			prompt_characteristics=prompt_characteristics,
			inference_batch_size=4,
			temperature=0.1, top_p=0.9, max_tokens=512, do_sample=True, num_beams=1,
			default_prediction=0,
			# OTHERS
			spec_str="_multimodal_exp",
			seed=seed,
			verbosity=False)
