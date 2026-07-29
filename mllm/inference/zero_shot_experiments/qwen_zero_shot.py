from constants import *

from inference.zero_shot_inference import zero_shot_inf


if __name__ == "__main__":
	csv_path = f"{DATA_DIR}/mobygaze_dataframe_soft_visual_corrected.csv"
	modality="visual"
	test_annotator = 'annotator_2'
	num_folds = NUM_FOLDS
	prompt_characteristics = {'exp_name': 'EN_HNS', 'spec': 'new'}

	zero_shot_inf(  # TEST DATASET
		csv_path=csv_path,modality=modality, test_annotator=test_annotator, num_folds=num_folds, folds_info=FOLDS_INFO,
		# SAMPLING PARAMETERS
		num_frames=8, max_pixels=64 * 28 * 28, align_factor=28,
		# MODEL PARAMETERS
		model_path= "Qwen/Qwen2.5-VL-7B-Instruct", max_model_len=4096, gpu_index=5,
		# INFERENCE PARAMETERS
		prompt_characteristics=prompt_characteristics,
		inference_batch_size=4,
		temperature=0.1, top_p=0.9, max_tokens=512, do_sample=True, num_beams=1,
		default_prediction=0,
		# OTHERS
		spec_str="_zero_shot",
		seed=42,
		verbosity=False)
