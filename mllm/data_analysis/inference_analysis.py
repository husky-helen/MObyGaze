import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import textwrap
import warnings
import random

from global_utils import get_shot_str, classif_info_from_exp_name
from getting_result_tables.get_results_tables import get_prediction_df
from dataset.dataset import MLLMDataset
from constants import *


warnings.filterwarnings("ignore")


def plot_from_txts_imgs(texts, imgs, spec_str, sup_title, spec_folder=None, text_ratio=1.5):
	n_texts = len(texts)

	# Increase height according to the amount of text
	fig = plt.figure(
		figsize=(15, 2 + 2.5 * n_texts),
		constrained_layout=True
	)

	gs = GridSpec(
		nrows=1 + n_texts,
		ncols=3,
		figure=fig,
		height_ratios=[4] + [text_ratio] * n_texts
	)

	# --------------------
	# Top row: 3 images
	# --------------------
	for i, img in enumerate(imgs):
		ax = fig.add_subplot(gs[0, i])
		ax.imshow(img)
		ax.axis("off")

	# --------------------
	# Text rows
	# --------------------
	for row, txt in enumerate(texts, start=1):
		ax = fig.add_subplot(gs[row, :])  # span all 3 columns
		ax.axis("off")

		wrapped = textwrap.fill(txt, width=120)

		ax.text(
			0.01, 0.95,
			wrapped,
			transform=ax.transAxes,
			va="top",
			ha="left",
			fontsize=12
		)

	fig.suptitle(
		sup_title,
		fontsize=18,
		fontweight="bold"
	)

	# Saving plots
	#plt.tight_layout()
	# Save
	plot_dir = f"{PROJECT_DIR}/plots/inference_analysis" + (f"/{spec_folder}" if spec_folder is not None else "")
	if not os.path.exists(plot_dir):
		os.makedirs(plot_dir)
	plt.savefig(f"{plot_dir}/{spec_str}.png", dpi=200)
	plt.close()
	print(f"{plot_dir}/{spec_str}.png saved")


def qualitative_plots(all_experiments, test_annotator, num_examples=30, spec_ids=None,
					  spec_folder=None):
	"""
	:param all_experiments: a dictionary of experiments, whose keys will be the name of each paragraph
	N.B. : the experiments should be consistent, ie. having the same test set
	:param test_annotator: the annotator to sample examples from
	:param num_examples: the number of examples randomly generated
	:param spec_ids: additionally to the num_examples, the specific ids to generate the plot for
	N.B. : it is simply the index of the result dataframe, so it is less practical than guids
	:param spec_folder: the folder to save the plots in
	"""

	model_names = list(all_experiments.keys())
	# 1. We load the inference results
	results_df = dict()
	for model_name, experiment in all_experiments.items():
		base_model_predictions = get_prediction_df(**experiment, test_annotator=test_annotator, inference_set='test')
		results_df[model_name] = base_model_predictions

	mllm_dataset = MLLMDataset(gt_csv=base_model_predictions, modality="visual", fold_idx=0,
							   folds_info={0: {"test": ALL_MOVIES, "validation": []}},
							   classif_info=classif_info_from_exp_name("EN_HNS"), split="test",
							   # classif_info plays no role here
							   annotator=test_annotator)
	# it does not matter which predictions the MLLMDataset is associated to ; it is only used to extract frames
	n = len(base_model_predictions)

	# 2. We select the guids of examples to print out
	if isinstance(num_examples, str) and num_examples == "all":
		samples = range(n)
	else:
		samples = random.sample(range(n), num_examples) + (spec_ids if spec_ids is not None else [])
	for id_example in samples:
		print(f"\nid {id_example}...")
		test_title, test_guid, test_label, imgs = mllm_dataset.get_clip_frames(idx=id_example, num_frames=3)
		texts = []
		for model_name in model_names:
			pred_row = results_df[model_name].iloc[id_example, :]
			txt_output = pred_row.output_text
			concepts = pred_row.concepts
			texts.append(f"{model_name}: {txt_output}")

		plot_from_txts_imgs(texts, imgs, spec_str=f"guid_{test_guid}",
							sup_title=f"{test_title} : {test_label} {concepts}",
							spec_folder=spec_folder)


if __name__ == "__main__":
	model_names = ["Qwen/Qwen2.5-VL-7B-Instruct", "OpenGVLab/InternVL3-8B"
	]
	annotator = 'annotator_2'
	prompt_characteristics = {'exp_name': 'EN_HNS', 'spec': 'new'}
	exp_name = 'EN_HNS'
	zero_shot = True
	preliminary_res_name = None  # f"mobygaze_dataframe_annotator_2_num1_demo_pool_for_EN_HNS"
	few_shot_annotator = annotator
	total_expert_budget = None  # 50
	expert_spec_str = ''
	pooling_type = 'base'
	selection_type = None
	few_shot_num = None
	few_shot_order = None

	all_experiments = dict()
	for model_name in model_names:
		short_model_name = model_name.split('/')[-1]
		shot_str = get_shot_str(zero_shot=zero_shot, pooling_type=pooling_type, few_shot_annotator=few_shot_annotator,
		                        selection_type=selection_type, total_expert_budget=total_expert_budget,
		                        expert_spec_str=expert_spec_str, few_shot_num=few_shot_num,
		                        few_shot_order=few_shot_order,
		                        preliminary_res_name=preliminary_res_name)

		experiment = {'model_path': model_name,
		              'prompt_characteristics': prompt_characteristics,
		              'shot_str': shot_str,
		              'exp_name': exp_name,
		              'dataset_name': 'mobygaze_dataframe_soft_visual_corrected',
		              'modality': 'visual',
		              'seed': 42,
		              'spec_str': "_zero_shot_1fps"}

		all_experiments[short_model_name] = experiment

	qualitative_plots(all_experiments=all_experiments, test_annotator='annotator_2', num_examples=30,
					  spec_ids=[130, 144, 749],
					  # correspond to guids [2462, 2499, 4692], ie.
					  # [Marley and Me beach, Marley and Me domestic scene, Millenium sex scene]
					  spec_folder=f"zero_shot_1fps_ENHN_S_{prompt_characteristics['spec']}")
	# Also works for few-shot, but for now it does not plot the few-shot examples
