import time

import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor, AutoTokenizer
import gc
import torch
from torch.utils.data import DataLoader

print(f"torch.__version__ : {torch.__version__}")
print(f"torch.version.cuda : {torch.version.cuda}")
print(f"torch.cuda.get_arch_list() : {torch.cuda.get_arch_list()}")
print(f"torch.cuda.get_device_name(0) : {torch.cuda.get_device_name(0)}")
print(f"torch.cuda.get_device_capability(0) : {torch.cuda.get_device_capability(0)}")
# device = torch.device("cpu")
import json
import gc
from functools import partial

from constants import *
from dataset.dataset import MLLMDataset
from global_utils import *
from inference.inference_utils import *


# Models are automatically saved in HuggingFace cache : to avoid overloading your homespace,
# first add this kind of line to your .bashrc file (and rerun it, for instance closing/reopening cluster) :
# export HF_HOME=/data1/paul/huggingface_cache


# Actual inference code
def few_shot_inf(  # TEST DATASET
		csv_path, modality="visual", test_annotator=None, num_folds=NUM_FOLDS, folds_info=FOLDS_INFO,
		# FEW-SHOT
		pooling_type=None, few_shot_annotator=None, selection_type=None, prompt_strategy=None,
		total_expert_budget=None, expert_spec_str=None, few_shot_num=None,
		few_shot_order=None,
		preliminary_res_name=None,
		feature_extractor="xclip",
		# SAMPLING PARAMETERS
		num_frames=16, num_frames_max=100, max_pixels=128 * 28 * 28, align_factor=28,
		# MODEL PARAMETERS
		model_path="Qwen/Qwen2.5-VL-7B-Instruct", max_model_len=4096, gpu_index=1,
		# INFERENCE PARAMETERS
		prompt_characteristics=None,
		inference_batch_size=4,
		temperature=0.1, top_p=0.9, max_tokens=1024, do_sample=True, num_beams=1,
		default_prediction=0,
		# OTHERS
		spec_str="",
		seed=42,
		verbosity=True,
		print_prompt=True,
		custom_str_conversion=None):
	"""
	:param csv_path: path of the dataset to perform inference on
	:param modality: if not None, we internally consider the CMSD version of the dataset according to this modality
	(in zero-shot, this is used when saving true labels).
	It can be 'visual', 'text', 'audio' or any sequence of modalities separated by '+' (for instance 'text+visual')
	:param test_annotator: the annotator to test the model on
	:param num_folds: iterable of the fold indices (which are the keys of folds_info) to consider
	:param folds_info: a dictionary describing the composition of folds (regarding imdbs)
	The expected structure is folds_info = {i_fold: {"test": list_of_test_imdbs, "validation": list_of_validation_imdbs}} ;
	see FOLDS_INFO as an example ; train_imdbs are defined as complementary of test and val imdbs

	:param pooling_type: a string describing the nature of the demonstration pool
	Possible choices are : "base", "expert_annotated", "preliminary_few_shot"
	In all case, when classifying a given sample from a given fold, the demonstration pool consists in
	all other folds of the demonstration dataset.
	"base" : demonstration pool = the initial dataset, with the initial annotations
	"expert_annotated" : demonstration pool = a small dataset (defined by density-based sampling) annotated by
	experts ; this choice requires the arguments total_expert_budget and expert_spec_str characterizing it
	"preliminary_few_shot": demonstration pool = in VIOLA fashion, the initial dataset with a small fraction being
	annotated by expert and the rest being labeled by a preliminary few-shot based on those expert labels
	:param few_shot_annotator: the annotator from which few-shot samples are selected
	:param selection_type: a string describing how few-shot examples are selected.
	Possible choices are : 'similarity', 'balanced_similarity', 'random'
	:param prompt_strategy: a dictionary describing the structure of the few-shot prompt
	:param total_expert_budget: if pooling_type="expert_annotated", indicates the budget if the expert-annotated dataset
	:param expert_spec_str: if pooling_type="expert_annotated", the string that characterizes the expert-annotated dataset
	:param few_shot_num: the number of few-shot demonstrations
	:param few_shot_order: the order in which presenting few-shot
	Possible choices are : "ground_truth_first", "ground_truth_last", "random"
	:param preliminary_res_name: if pooling_type="preliminary_few_shot", the string that characterizes the few-shot dataset
	:param feature_extractor: the features to use for computing similarity between test and demonstration samples
	Possible choices are : 'xclip'

	:param num_frames: either an int indicating the number of frames to sample for each clip, or "one_per_second"
	:param num_frames_max: if num_frames="one_per_second", an int indicating the upper_bound of the numer of frames
	:param max_pixels: for compression, the maximum number of pixels
	:param align_factor: for compression, we ensure dimensions are multiples of align_factor
	:param model_path: path of the HuggingFace model
	Supported models are "OpenGVLab/InternVideo2_5_Chat_8B", "OpenGVLab/InternVL3-8B", "DAMO-NLP-SG/VideoLLaMA3-7B",
	"Qwen/Qwen2.5-VL-7B-Instruct"
	:param max_model_len: maximum input sequence length (used for Qwen model only)
	:param gpu_index: NOT USED CURRENTLY gpu_index to run the model on
	:param prompt_characteristics: a dictionary describing the prompt ; it should contain keys 'exp_name' and 'spec'
	:param inference_batch_size: inference batch size (must be 1 for all other models than Qwen)
	:param temperature: temperature used in sampling during output generation
	:param top_p: top_p used in sampling during output generation
	(instead of all tokens, consider only the top tokens that together make up top_p probability)
	:param max_tokens: max number of generated tokens
	:param do_sample: if generations is performed using sampling
	:param num_beams: NOT USED CURRENTLY number of beams in case of beam search during output generation
	:param default_prediction: integer default prediction in case the answer does not respect the template and
	is not supported by recovery strategies
	:param spec_str: a string characterizing the experiment (to avoid overwriting)
	:param seed: random seed for reproducibility
	:param verbosity: verbosity boolean
	:param print_prompt: whether we print the formatted prompt
	:param custom_str_conversion: a way to custom the conversion from text output to integer
	It should be consistent with the specified prompt. If None, the code uses the default conversion, expecting the
	positive classes to be 'Normal' and 'Objectifying'.
	"""

	model_str = model_path.split('/')[-1]
	dataset_name = csv_path.split('/')[-1][:(-4)]
	shot_str = get_shot_str(zero_shot=False, pooling_type=pooling_type, few_shot_annotator=few_shot_annotator,
							selection_type=selection_type, total_expert_budget=total_expert_budget,
							expert_spec_str=expert_spec_str, few_shot_num=few_shot_num,
							few_shot_order=few_shot_order,
							preliminary_res_name=preliminary_res_name)

	setup_seed(seed)

	exp_name = prompt_characteristics['exp_name']
	classif_info = classif_info_from_exp_name(exp_name)
	full_exp_name = f"{prompt_characteristics['exp_name']}_{prompt_characteristics['spec']}" # for saving results
	str_conversion = str_conversion_from_exp_name(exp_name) if custom_str_conversion is None else custom_str_conversion
	extracting_answers = partial(answer_from_text_output, str_conversion=str_conversion)

	print(f"\n## MODEL {model_path}\n")
	if model_path != "Qwen/Qwen2.5-VL-7B-Instruct":
		assert inference_batch_size == 1

	if model_path == "OpenGVLab/InternVideo2_5_Chat_8B":
		print(f"\nWarning : with model OpenGVLab/InternVideo2_5_Chat_8B, some values of num_frames simply fail. "
			  f"num_frames=8,16 work\n")
		assert num_frames != "one_per_second"
		# For some reason, some values of num_frames (like 14) are not supported by this model
		# (I suspect is a divisibility problem)
		# So, even integer num_frames might fail, but with adaptable number of frames using "one_per_second",
		# it is sure that for some clip some num_frames will fail

	# MODEL
	llm, tokenizer, image_processor, sampling_params = load_model(model_path, max_model_len=max_model_len,
	                                                              temperature=temperature, top_p=top_p,
	                                                              max_tokens=max_tokens, do_sample=do_sample,
	                                                              num_beams=num_beams,
																  few_shot_num=few_shot_num)
	print(f"\nModel loaded successfully!\n")

	# PROMPT
	prompt = prompt_from_characteristics(prompt_characteristics)
	if verbosity:
		print(f"prompt :\n{prompt}\n")

	for fold_idx in num_folds:
		print(f"\n\nFOLD {fold_idx}...\n")
		# DATASET
		if pooling_type == "base":
			csv_directory = f"{PROJECT_DIR}/data/demo_pool_from_base"
			demo_path = f"{csv_directory}/{dataset_name}_demo_pool_for_{exp_name}.csv"
		elif pooling_type == "expert_annotated":
			csv_directory = f"{PROJECT_DIR}/data/density_based_sampling"
			demo_path = (
				f"{csv_directory}/{dataset_name}_{test_annotator}_{exp_name}_{total_expert_budget}_{expert_spec_str}.csv")
		elif pooling_type == 'preliminary_few_shot':
			csv_directory = f"{PROJECT_DIR}/data/demo_pool_from_few_shot"
			demo_path = f"{csv_directory}/{preliminary_res_name}.csv"
		else:
			raise Exception(f"pooling_type {pooling_type} is not supported for now.")

		demonstration_dataset = MLLMDataset(gt_csv=demo_path, modality=modality, fold_idx=fold_idx, folds_info=folds_info,
		                           classif_info=classif_info, split="train", # train fold = all other test folds
		                           annotator=few_shot_annotator,
		                           num_frames=num_frames, num_frames_max=num_frames_max, max_pixels=max_pixels, align_factor=align_factor,
		                           seed=seed,
											feature_extractor=feature_extractor)
		print(f"len(demonstration_dataset) : {len(demonstration_dataset)}")
		test_dataset = MLLMDataset(gt_csv=csv_path,modality=modality, fold_idx=fold_idx, folds_info=folds_info,
		                           classif_info=classif_info, split="test",
		                           annotator=test_annotator,
		                           num_frames=num_frames, num_frames_max=num_frames_max, max_pixels=max_pixels, align_factor=align_factor,
		                           seed=seed,
								   # FEW-SHOT INFERENCE
		                           demonstration_dataset=demonstration_dataset,
								   selection_type=selection_type, few_shot_num=few_shot_num,
								   few_shot_order=few_shot_order,
								   feature_extractor=feature_extractor
								   )
		print(f"len(test_dataset) : {len(test_dataset)}")
		test_loader = DataLoader(test_dataset, batch_size=inference_batch_size, shuffle=False, num_workers=16,
		                         collate_fn=few_shot_qwen_collate_fn)

		# To help managing memory
		print(f"\nallocated: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
		print(f"\nreserved: {torch.cuda.memory_reserved() / 1024 ** 3} GB") # equal to the max memory it had to use so far

		# INFERENCE
		# Thanks to shuffle=False, we will be able to store test probabilities directly
		# (without having to recompute the inference order)
		log_txts = []
		list_preds = []
		list_binary_targets = []
		list_fs_label_strs = []
		list_fs_guids = []
		t_start = time.time()
		with ((torch.no_grad())):  # not necessary with vllm
			for i_test, tbatch in enumerate(test_loader):
				if verbosity:
					print(f"batch {i_test}...")

				t_batch_start = time.time()
				inputs, clip_labels, one_hot_concepts, guids, few_shot_label_strs, few_shot_concepts, few_shot_guids = tbatch

				text_outputs = batch_inference_fs(inputs, llm, tokenizer, image_processor, prompt, sampling_params,
				                               model_path,
				                               num_frames, few_shot_label_strs, few_shot_concepts, prompt_strategy,
												  print_prompt=(print_prompt and i_test==0))

				t_batch_end = time.time()

				answers = list(map(extracting_answers, text_outputs))
				preds = predictions_from_answers(answers, str_conversion)
				# We update the full results
				log_txts += text_outputs
				list_preds.append(preds)
				list_binary_targets.append(clip_labels.detach().cpu().numpy())
				list_fs_label_strs += few_shot_label_strs
				list_fs_guids += few_shot_guids

				#gc.collect()
				#torch.cuda.empty_cache()

		preds_np = np.concatenate(list_preds, axis=0)
		binary_targets_np = np.concatenate(list_binary_targets, axis=0)
		# Predictions may contain None
		original_preds = preds_np.copy()
		count_none = np.sum(preds_np == None)
		print(f"\nThere are initially {count_none} / {len(preds_np)} = {round(count_none / len(preds_np) * 100, 2)} % "
		      f"None predictions (ie. the answer template was not respected).")
		# We replace them by default_prediction
		preds_np[preds_np == None] = default_prediction
		preds_np = preds_np.astype(np.int_) # if the array contained None, it was of type object, while we want int
		t_end = time.time()
		print(f"\nTime for full test fold inference : {round(t_end - t_start, 2)} s -> "
			  f"{round((t_end - t_start)/len(test_dataset), 2)} s per test example")

		# Now we can compute performances and save results
		test_hard_metrics = get_hard_metrics(binary_targets_np, preds_np)
		print(f"\n{test_hard_metrics}")

		# For finer processing of results later, we also save the per-observation results, in a similar fashion to infer.py
		original_df = test_dataset.gt_df # we cannot use get_specific_df because the CMSD dataset in gt_df may be shuffled compared to the original df
		original_df["fold"] = 0
		original_df = get_test_inferred_df(original_df, preds_np, original_preds, log_txts,
										   list_fs_label_strs, list_fs_guids)

		csv_directory = f"{PROJECT_DIR}/inference_results/{full_exp_name}/{shot_str}/{model_str}/{dataset_name}_{modality}"
		if not os.path.exists(csv_directory):
			os.makedirs(csv_directory)
		combined_csv_path = f"{csv_directory}/test_results_seed_{seed}_{test_annotator}_{exp_name}_fold_{fold_idx}_{spec_str}.csv"
		original_df.to_csv(combined_csv_path, sep=';', index=False)
		print(f"\nSaved successfully test inference per-observation results at {combined_csv_path}")
