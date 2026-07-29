from constants import *

from inference.few_shot_inference import few_shot_inf


if __name__ == "__main__":
	csv_path = f"{DATA_DIR}/mobygaze_dataframe_soft_visual_corrected.csv" # f"{DATA_DIR}/mobygaze_dataframe_soft_visual_corrected.csv" # f"{DATA_DIR}/very_small_dataset.csv"
	modality="visual"
	test_annotator = 'annotator_2'
	num_folds = NUM_FOLDS
	gpu_index = 5 # N.B. : command without effect for now
	prompt_characteristics = {'exp_name': 'ENHN_S', 'spec': 'small'}

	prompt_strategy = {
		# The prompt preceding few-shot examples
		'global_few_shot_prompt': "Below, you will find example videos and corresponding desired answers.",
		# The prompt preceding the core example, only used if examples_first
		'global_core_prompt': "Here is the actual example to classify.",
		# Placement of video tokens compared to description text tokens, for few-shot examples (for the core example we follow the doc template)
		'video_placement': 'above',  # 'above', 'below'
		# Placement of few-shot examples compared to the core example (the task prompt is always at the beginning anyway)
		'examples_order': 'examples_first',  # 'examples_last', 'examples_first'
		# To what extent we describe few-shot examples
		'description': 'label+concepts'  # 'none', 'label', 'concepts', 'label+concepts'
	}

	few_shot_inf(  # TEST DATASET
		csv_path=csv_path,modality=modality, test_annotator=test_annotator, num_folds=num_folds, folds_info=FOLDS_INFO,
		# FEW-SHOT
		pooling_type='base', few_shot_annotator=test_annotator, selection_type='balanced_similarity', prompt_strategy=prompt_strategy,
		total_expert_budget=None, expert_spec_str=None, few_shot_num=4,
		few_shot_order='random',
		preliminary_res_name=None,
		feature_extractor="xclip",
		# SAMPLING PARAMETERS
		num_frames=8, max_pixels=64 * 28 * 28, align_factor=28,
		# MODEL PARAMETERS
		model_path= "OpenGVLab/InternVideo2_5_Chat_8B", max_model_len=8192, gpu_index=5,
		# INFERENCE PARAMETERS
		prompt_characteristics=prompt_characteristics,
		inference_batch_size=1,
		temperature=0.1, top_p=0.9, max_tokens=1024, do_sample=True, num_beams=1,
		default_prediction=0,
		# OTHERS
		spec_str="_examples_first",
		seed=42,
		verbosity=False)
