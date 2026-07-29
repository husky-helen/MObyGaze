import numpy as np
import matplotlib.pyplot as plt
import random

from constants import *
from dataset.dataset import MLLMDataset
from global_utils import *


def movie_sampling_analysis(  # TEST DATASET
		csv_path, modality="visual", imdbs_to_test=None,
		# FEW-SHOT
		few_shot_annotator=None,
		# SAMPLING PARAMETERS
		num_frames=16, max_pixels=128 * 28 * 28, align_factor=28,
		# MODEL PARAMETERS
		prompt_characteristics=None,
		seed=42,
		feature_extractor='xclip',
		num_examples = 3,
		verbosity=True):
	"""
	model_path: name of the Huggingface model
	"""

	dataset_name = csv_path.split('/')[-1][:(-4)]
	setup_seed(seed)

	exp_name = prompt_characteristics['exp_name']
	classif_info = classif_info_from_exp_name(exp_name)
	str_conversion = str_conversion_from_exp_name(exp_name)

	top_k = 10

	for imdb_k in imdbs_to_test:
		print(f"\n{imdb_k}...")
		imdb_dataset = MLLMDataset(gt_csv=csv_path, modality=modality, fold_idx=0,
		                           folds_info={0: {"test": [imdb_k], "validation": []}},
		                           classif_info=classif_info, split="test",  # train fold = all other test folds
		                           annotator=few_shot_annotator,
		                           num_frames=num_frames, max_pixels=max_pixels, align_factor=align_factor,
		                           seed=seed, precompute_features=False,
								   feature_extractor=feature_extractor)
		n = len(imdb_dataset)
		samples = random.sample(range(n), num_examples)
		for id_example in samples:
			test_title, test_guid, test_label, test_frames = imdb_dataset.get_clip_frames(idx=id_example,
																						  num_frames=num_frames,
																						  max_pixels=max_pixels, factor=align_factor)
			frame_shape = test_frames[0].shape
			print(f"frame_shape : {frame_shape} -> {frame_shape[0] * frame_shape[1]} pixels (vs. {max_pixels} max_pixels)")

			fig, axes = plt.subplots(1, num_frames, figsize=(12, 8))
			# First row: Movie A
			for i, frame in enumerate(test_frames):
				axes[i].imshow(frame)
				#axes[i].axis("off")

			plt.tight_layout()
			# Save
			plot_dir = f"{PROJECT_DIR}/plots/sampling_analysis"
			if not os.path.exists(plot_dir):
				os.makedirs(plot_dir)
			plt.savefig(f"{plot_dir}/{imdb_k}_{id_example}_in_{max_pixels}_{align_factor}.png", dpi=200)
			plt.close()
			print(f"{plot_dir}/{imdb_k}_{id_example}_in_{max_pixels}_{align_factor}.png saved")


if __name__ == "__main__":
	csv_path = f"{DATA_DIR}/mobygaze_dataframe.csv" #  # f"{DATA_DIR}/very_small_dataset.csv"
	modality = "visual"
	test_annotator = 'annotator_2'
	num_folds = NUM_FOLDS
	prompt_characteristics = {'exp_name': 'ENHN_S', 'spec': 'small'}
	feature_extractor = 'new_xclip'

	movie_sampling_analysis(  # TEST DATASET
		csv_path=csv_path, modality=modality, imdbs_to_test=ALL_MOVIES,
		# FEW-SHOT
		few_shot_annotator=test_annotator,
		# SAMPLING PARAMETERS
		num_frames=2, max_pixels=64 * 28 * 28, align_factor=28, # max_pixels=64 * 28 * 28, align_factor=28,
		# INFERENCE PARAMETERS
		prompt_characteristics=prompt_characteristics,
		seed=42,
		verbosity=True,
		feature_extractor=feature_extractor)
