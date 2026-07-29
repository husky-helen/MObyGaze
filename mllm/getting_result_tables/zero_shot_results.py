import warnings

from getting_result_tables.get_results_tables import main_performances
from global_utils import get_shot_str

if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	# First, we have to keep track of all the experiments we've done
	all_experiments = dict()  # will be a dictionary of experiments dictionaries
	# N.B. : for the sake of simplicity, an experiment dictionary may contain useless arguments
	# (for instance contrastive arguments for non-contrastive models) but that shouldn't have any consequence
	# N.B. : Be careful to let the actual model_name at the end of the all_experiments keys
	ANNOTATOR = 'annotator_2'
	prompt_characteristics = {'exp_name': 'EN_HNS', 'spec': 'new'}
	# FEW-SHOT ARGUMENTS (useless here, except zero_shot=True)
	zero_shot = True
	preliminary_res_name = None
	few_shot_annotator = ANNOTATOR
	total_expert_budget = None
	expert_spec_str = ''
	pooling_type = 'base'
	selection_type = None
	few_shot_num = None
	few_shot_order = None

	for model_name in ["DAMO-NLP-SG/VideoLLaMA3-7B", "lmms-lab/LLaVA-Video-7B-Qwen2", "Qwen/Qwen2.5-VL-7B-Instruct",
					   "OpenGVLab/InternVideo2_5_Chat_8B", "OpenGVLab/InternVL3-8B"]:
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
							'modality': 'visual',
							'seed': 42,
							'spec_str': "_zero_shot"}
		all_experiments[model_name] = first_experiment

	main_performances(all_experiments, test_annotators=[ANNOTATOR], saving_str="_zero_shot_EN_HNS")
