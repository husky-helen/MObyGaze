import torch
import os
import random
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
	average_precision_score
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor, AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
	DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import copy
import re
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from safetensors.torch import load_file

from constants import *


def qwen_collate_fn(batch):
	inputs = [{
				"multi_modal_data": {"video": [item[0]]}
			} for item in batch]
	clip_labels = torch.tensor([item[1] for item in batch])
	one_hot_concepts = torch.stack([item[2] for item in batch])
	guids = np.stack([item[3] for item in batch])
	subtitles = [item[4] for item in batch]

	return inputs, clip_labels, one_hot_concepts, guids, subtitles


def few_shot_qwen_collate_fn(batch):
	inputs = [{
				"multi_modal_data": {"video": [item[0]] + item[4]}
			} for item in batch]
	clip_labels = torch.tensor([item[1] for item in batch])
	one_hot_concepts = torch.stack([item[2] for item in batch])
	guids = np.stack([item[3] for item in batch])
	few_shot_label_strs = [item[5] for item in batch]
	few_shot_concepts = [item[6] for item in batch]
	few_shot_guids = [item[7] for item in batch]

	return inputs, clip_labels, one_hot_concepts, guids, few_shot_label_strs, few_shot_concepts, few_shot_guids


def load_peft_model(base_model_path: str, checkpoint_dir: str):
	print(f"Loading base model: {base_model_path}")
	model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
		base_model_path, torch_dtype=torch.bfloat16, device_map="cuda",
		attn_implementation="sdpa",
		# NOTE: flash_attention_2 hits a rotary-embedding dtype mismatch
		# specifically on generate()'s KV-cache/incremental-decoding path
		# (a different code path than the full-sequence training forward
		# pass, where FA2 works fine). sdpa avoids this and doesn't need
		# FA2's memory savings here -- no large cross-entropy logits tensor
		# during generation, unlike training.
	)

	if checkpoint_dir != '':
		print(f"Applying LoRA adapter from: {checkpoint_dir}")
		model = PeftModel.from_pretrained(model, checkpoint_dir)

		# Defensive: also load the connector (visual.merger) weights explicitly
		# from whatever file in checkpoint_dir has them, regardless of whether
		# PEFT's own adapter-loading path picked them up. This is the runtime
		# counterpart of the [Save check] added in sft_video.py.

		found_merger = False
		for fname in os.listdir(checkpoint_dir):
			if fname.endswith(".safetensors"):
				state = load_file(os.path.join(checkpoint_dir, fname))
				# TODO: remove verbosity:
				print(f"\n{fname} keys :")
				for k in state:
					print(k)
				merger_keys = {k: v for k, v in state.items() if "merger" in k}
				if merger_keys:
					missing, unexpected = model.load_state_dict(merger_keys, strict=False)
					found_merger = True
					print(f"[Load check] Loaded {len(merger_keys)} merger keys from {fname}")
		if not found_merger:
			print("[Load check] WARNING: no merger keys found in any checkpoint file -- "
				  "connector is at its PRETRAINED state, not fine-tuned. Results will not "
				  "reflect connector training. Fix the save path before trusting this eval.")

	model.eval()
	return model


def load_model(model_path, max_model_len=4096,
			   temperature=0.1, top_p=0.9, max_tokens=1024,
			   do_sample=False, num_beams=1, few_shot_num=0,
			   modality='visual', peft_checkpoint_dir=None):
	if model_path == "Qwen/Qwen2.5-VL-7B-Instruct":
		if peft_checkpoint_dir is not None:
			llm = load_peft_model(model_path, peft_checkpoint_dir)
		else:
			llm = LLM(
				model=model_path,
				trust_remote_code=True,
				dtype="float16",
				max_model_len=max_model_len,
				mm_processor_kwargs={"fps": 1,},
				limit_mm_per_prompt={"video": ((1+few_shot_num) if "visual" in modality else 0)},
				gpu_memory_utilization=0.8,
			)
		tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
		image_processor = None

	elif model_path == "lmms-lab/LLaVA-Video-7B-Qwen2":
		tokenizer, llm, image_processor, max_length = load_pretrained_model(model_path=model_path,
																			model_base=None, model_name=model_path,
		                                                                    attn_implementation='eager',
		                                                                    torch_dtype="bfloat16",
		                                                                    device_map="auto")
		llm.eval()
		# To avoid any problem linked to pad token (apparently originally none)
		tokenizer.add_special_tokens({'pad_token': '[PAD]'})
		llm.resize_token_embeddings(len(tokenizer))
		# tokenizer.pad_token = tokenizer.eos_token

	elif model_path == "DAMO-NLP-SG/VideoLLaMA3-7B":
		llm = AutoModelForCausalLM.from_pretrained(
			model_path,
			trust_remote_code=True,
			device_map="auto",
			torch_dtype=torch.float32,  # torch.float16 eventually causes float16 overflow
			attn_implementation="eager",
		)
		tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
		image_processor = None

	elif model_path in ["OpenGVLab/InternVideo2_5_Chat_8B", "OpenGVLab/InternVL3-8B"]:
		llm = AutoModel.from_pretrained(
			model_path,
			torch_dtype=torch.bfloat16,
			# low_cpu_mem_usage=True,
			use_flash_attn=False,
			trust_remote_code=True).eval().cuda()
		tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
		image_processor = None

	else:
		raise Exception(f"Model {model_path} is not supported.")

	sampling_params = dict(
		temperature=temperature, # from 0 to 1 ; the closer the 0, the sharper/more reproducible
		max_new_tokens=max_tokens, # max number of generated tokens
		top_p=top_p, # instead of all tokens, consider only the top tokens that together make up top_p probability
		do_sample=do_sample, # False:greedy/deterministic ; True: sampling according to the distribution
		num_beams=num_beams # if beam search (ie. instead of keeping only one partial sentence while generating, the model keeps several candidate continuations)
		# top_k : keeps only the top_k tokens (in modern setups, top_p is preferred)
	)

	return llm, tokenizer, image_processor, sampling_params


# util for Intern inference
def get_intern_inputs_from_frames(base_frames):
	pixel_values_list, num_patches_list = [], []
	transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
	                       T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
	                       T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
	for frame_index, frame in enumerate(base_frames):
		img = Image.fromarray(frame)
		pixel_values = transform(img)
		pixel_values = torch.stack([pixel_values])  # [1, 3, 448, 448]
		num_patches_list.append(pixel_values.shape[0])  # 1
		pixel_values_list.append(pixel_values)
	pixel_values = torch.cat(pixel_values_list)
	return pixel_values, num_patches_list


def batch_inference(inputs, llm, tokenizer, image_processor, prompt, subtitles, sampling_params, model_path,
					num_frames,
					modality,
					print_prompt=False,
					peft_checkpoint_dir=None  # simply used to know what kind of model we are facing
					):
	if model_path == "Qwen/Qwen2.5-VL-7B-Instruct" and peft_checkpoint_dir is not None:
		text_prompts = []
		for k in range(len(inputs)):
			final_prompt = prompt + subtitles[k] if ("text" in modality) else prompt
			messages = [
				{"role": "system", "content": SYSTEM_PROMPT},
				{
					"role": "user",
					"content": [{"type": "text", "text": final_prompt}] + (
						[{"type": "video", "video": "dummy_video.mp4",
						  "nframes": num_frames}] if "visual" in modality else [])
				}
			]
			# Now we can apply the chat template
			text_prompt = tokenizer.apply_chat_template(messages,
			                                            tokenize=False,
			                                            add_generation_prompt=True)
			text_prompts.append(text_prompt)
			if print_prompt and k== 0:
				print(f"\ntokenized prompt:\n{text_prompt}\n")

		videos = [input_k["multi_modal_data"]["video"] for input_k in inputs]
		inputs = tokenizer(text=text_prompts, videos=videos, padding=True, return_tensors="pt")
		inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
		result = llm.generate(**inputs,
		                      temperature=sampling_params['temperature'],
		                      top_p=sampling_params['top_p'],
		                      max_new_tokens=sampling_params['max_new_tokens'],
		                      do_sample=sampling_params['do_sample'],
		                      num_beams=sampling_params['num_beams'],
		                      )
		raw_text_outputs = tokenizer.batch_decode(result, skip_special_tokens=True)
		text_outputs = [raw_text.split('assistant\n')[1] for raw_text in raw_text_outputs]
	
	elif model_path == "Qwen/Qwen2.5-VL-7B-Instruct":
		# We update inputs with the common prompt
		for k in range(len(inputs)):
			final_prompt = prompt + subtitles[k] if ("text" in modality) else prompt
			messages = [
				{"role": "system", "content": SYSTEM_PROMPT},
				{
					"role": "user",
					"content": [{"type": "text", "text": final_prompt}] + ([{"type": "video", "video": "dummy_video.mp4", "nframes": num_frames}] if "visual" in modality else [])
				}
			]
			# Now we can apply the chat template
			text_prompt = tokenizer.apply_chat_template(messages,
												   tokenize=False,
												   add_generation_prompt=True)
			inputs[k]["prompt"] = text_prompt
			if not "visual" in modality:
				inputs[k]["multi_modal_data"]["video"] = []

			if print_prompt and k== 0:
				print(f"\ntokenized prompt:\n{text_prompt}\n")

		# Sampling params
		qwen_sampling_params = SamplingParams(
			temperature=sampling_params['temperature'],
			top_p=sampling_params['top_p'],
			max_tokens=sampling_params['max_new_tokens'],
		)
		
		result = llm.generate(inputs, sampling_params=qwen_sampling_params)
		# The standard behaviour of vllm is to do sampling ; greedy decoding is performed with temperature=0.
		# Beam search is a separate function.
		# It is alright since my defaults are do_sample=True and beam_search=1 ; if that changes, I should adapt that.
		text_outputs = [result[i_output].outputs[0].text for i_output in range(len(result))]

	elif model_path == "lmms-lab/LLaVA-Video-7B-Qwen2":
		# We generate the formatted prompt
		question = DEFAULT_IMAGE_TOKEN + prompt  # f"\n{time_instruction}\n{prompt}"
		conv = copy.deepcopy(conv_templates["qwen_1_5"])
		# a Conversation object, include in particular a .message field :
		# messages = [ [role, message pair], [role, message pair]]
		conv.append_message(conv.roles[0], question)
		conv.append_message(conv.roles[1], None)
		text_prompt = conv.get_prompt()
		if print_prompt:
			print(f"\ntokenized prompt:\n{text_prompt}\n")

		frames = inputs[0]["multi_modal_data"]["video"][0]
		video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
		input_ids = torch.stack([tokenizer_image_token(text_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")], dim=0).to("cuda")
		cont = llm.generate(
			input_ids,
			images=[video],
			modalities=["video"],
			temperature=sampling_params['temperature'],
			top_p=sampling_params['top_p'],
			max_new_tokens=sampling_params['max_new_tokens'],
			do_sample=sampling_params['do_sample'],
			num_beams=sampling_params['num_beams'],
		)
		text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

	elif model_path == "DAMO-NLP-SG/VideoLLaMA3-7B":
		frames = inputs[0]["multi_modal_data"]["video"][0]
		#print(f"frames.shape : {frames.shape}")
		conversation = [
			{"role": "system", "content": SYSTEM_PROMPT},
			{
				"role": "user",
				"content": [
					*[
						{"type": "image", "image": frame}
						for frame in frames
					],
					{"type": "text", "text": prompt},
				],
			},
		]
		llama_inputs = tokenizer(conversation=conversation, return_tensors="pt")
		llama_inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in llama_inputs.items()}
		if "pixel_values" in llama_inputs:
			llama_inputs["pixel_values"] = llama_inputs["pixel_values"].to(torch.float32)
		if print_prompt:
			text_prompt = tokenizer.batch_decode(llama_inputs['input_ids'], skip_special_tokens=True)[0]
			print(f"\ntokenized prompt:\n{text_prompt}\n")
		output_ids = llm.generate(**llama_inputs,
		                          temperature=sampling_params['temperature'],
		                          top_p=sampling_params['top_p'],
		                          max_new_tokens=sampling_params['max_new_tokens'],
		                          do_sample=sampling_params['do_sample'],
		                          num_beams=sampling_params['num_beams'],
		                          )
		text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

	elif model_path in ["OpenGVLab/InternVideo2_5_Chat_8B", "OpenGVLab/InternVL3-8B"]:
		frames = inputs[0]["multi_modal_data"]["video"][0]
		pixel_values, num_patches_list = get_intern_inputs_from_frames(frames)
		pixel_values = pixel_values.to(torch.bfloat16).to(llm.device)
		video_prefix = "".join([f"Frame{i + 1} of video 1 : <image>\n" for i in range(len(num_patches_list))])
		total_prompt = video_prefix + prompt
		if print_prompt:
			print(f"\ntokenized prompt:\n{total_prompt}\n")
		text_output, chat_history = llm.chat(tokenizer, pixel_values, total_prompt, sampling_params,
		                                     num_patches_list=num_patches_list, history=None, return_history=True)
		text_outputs = [text_output]

	else:
		raise Exception(f"Model {model_path} is not supported.")

	return text_outputs


def batch_inference_fs(inputs, llm, tokenizer, image_processor, prompt, sampling_params, model_path,
					num_frames, few_shot_label_strs, few_shot_concepts, prompt_strategy,
					   print_prompt=False):

	if model_path == "Qwen/Qwen2.5-VL-7B-Instruct":
		# We create the formatted prompts :
		for k in range(len(inputs)):
			core_frames = inputs[k]["multi_modal_data"]["video"][0]  # a np array
			fs_frames = inputs[k]["multi_modal_data"]["video"][1:]  # a list of np arrays

			# If we do examples first, we actually have to reorder the inputs :
			if prompt_strategy['examples_order'] == "examples_first":
				inputs[k]["multi_modal_data"]["video"] = fs_frames + [core_frames]

			text_prompt = get_few_shot_prompt(core_frames, fs_frames, tokenizer, prompt, model_path,
					num_frames, few_shot_label_strs[k], few_shot_concepts[k], prompt_strategy)

			inputs[k]["prompt"] = text_prompt

		if print_prompt:
			print(f"\ntokenized prompt:\n{text_prompt}\n")

		# Sampling params
		qwen_sampling_params = SamplingParams(
			temperature=sampling_params['temperature'],
			top_p=sampling_params['top_p'],
			max_tokens=sampling_params['max_new_tokens'],
		)

		result = llm.generate(inputs, sampling_params=qwen_sampling_params)
		# The standard behavior of vllm is to do sampling ; greedy decoding is performed with temperature=0.
		# Beam search is a separate function.
		# It is alright since my defaults are do_sample=True and beam_search=1 ; if that changes, I should adapt that.
		text_outputs = [result[i_output].outputs[0].text for i_output in range(len(result))]

	else:
		core_frames = inputs[0]["multi_modal_data"]["video"][0]  # a np array
		fs_frames = inputs[0]["multi_modal_data"]["video"][1:] # a list of np arrays

		if model_path == "DAMO-NLP-SG/VideoLLaMA3-7B":
			conversation = get_few_shot_prompt(core_frames, fs_frames, tokenizer, prompt, model_path,
			                                  num_frames, few_shot_label_strs[0], few_shot_concepts[0], prompt_strategy)
			llama_inputs = tokenizer(conversation=conversation, return_tensors="pt")
			llama_inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in llama_inputs.items()}
			if "pixel_values" in llama_inputs:
				llama_inputs["pixel_values"] = llama_inputs["pixel_values"].to(torch.float32)
			if print_prompt:
				text_prompt = tokenizer.batch_decode(llama_inputs['input_ids'], skip_special_tokens=True)[0]
				print(f"\ntokenized prompt:\n{text_prompt}\n")
			output_ids = llm.generate(**llama_inputs,
									  temperature=sampling_params['temperature'],
									  top_p=sampling_params['top_p'],
									  max_new_tokens=sampling_params['max_new_tokens'],
									  do_sample=sampling_params['do_sample'],
									  num_beams=sampling_params['num_beams'],
									  )
			text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

		elif model_path in ["OpenGVLab/InternVideo2_5_Chat_8B", "OpenGVLab/InternVL3-8B"]:
			# The base input for zero-shot
			pixel_values, num_patches_list = get_intern_inputs_from_frames(core_frames)
			pixel_values = pixel_values.to(torch.bfloat16).to(llm.device)
			# And for few-shot examples:
			pixel_values_fs_list = []
			num_patches_list_fs_list = []
			for i_example, fs_video in enumerate(fs_frames):
				pixel_values_fs, num_patches_list_fs = get_intern_inputs_from_frames(fs_video)
				pixel_values_fs = pixel_values_fs.to(torch.bfloat16).to(llm.device)
				# We update the video representations by appending the frames of the few-shot example
				pixel_values_fs_list.append(pixel_values_fs)
				num_patches_list_fs_list += num_patches_list_fs

			total_prompt = get_few_shot_prompt(core_frames, fs_frames, tokenizer, prompt, model_path,
			                                  num_frames, few_shot_label_strs[0], few_shot_concepts[0], prompt_strategy)
			if print_prompt:
				print(f"\ntokenized prompt:\n{total_prompt}\n")

			# If we do examples first, we actually have to reorder the inputs :
			if prompt_strategy['examples_order'] == "examples_last":
				pixel_values = torch.cat([pixel_values]+pixel_values_fs_list)
				num_patches_list = num_patches_list + num_patches_list_fs_list
			elif prompt_strategy['examples_order'] == "examples_first":
				pixel_values = torch.cat(pixel_values_fs_list+[pixel_values])
				num_patches_list = num_patches_list_fs_list + num_patches_list
			else:
				raise Exception(f"examples_order {prompt_strategy['examples_order']} is not supported.")

			text_output, chat_history = llm.chat(tokenizer, pixel_values, total_prompt, sampling_params,
												 num_patches_list=num_patches_list, history=None, return_history=True)
			text_outputs = [text_output]

		else:
			raise Exception(f"Model {model_path} is not supported.")

	return text_outputs


def get_specific_few_shot_prompts(few_shot_label_strs_example, few_shot_concepts_example, prompt_strategy):
	specific_few_shot_prompts = []
	video_placement = prompt_strategy["video_placement"]
	for i_demo, (demo_label_str, demo_concepts) in enumerate(zip(few_shot_label_strs_example, few_shot_concepts_example)):
		if prompt_strategy['description'] == "none":
			specific_few_shot_prompts.append("")
		else:
			demo_text = f"The video {video_placement} is <answer>[INSERT_LABEL]</answer>. ".replace("[INSERT_LABEL]",demo_label_str) if "label" in prompt_strategy['description'] else ""
			if "concept" in prompt_strategy['description']:
				if demo_concepts == []:
					demo_text += "It contains no objectifying concept."
				else:
					demo_concepts_str = ", ".join(demo_concepts)
					demo_text += f"More precisely, it contains the following concepts : [INSERT_CONCEPTS].".replace(
						"[INSERT_CONCEPTS]", demo_concepts_str)
			specific_few_shot_prompts.append(demo_text)
	return specific_few_shot_prompts


def get_few_shot_prompt(core_frames, fs_frames, tokenizer, prompt, model_path,
					num_frames, few_shot_label_strs_example, few_shot_concepts_example, prompt_strategy):

	global_few_shot_prompt = prompt_strategy['global_few_shot_prompt']
	global_core_prompt = prompt_strategy['global_core_prompt']
	specific_few_shot_prompts = get_specific_few_shot_prompts(few_shot_label_strs_example, few_shot_concepts_example, prompt_strategy)

	if model_path == "Qwen/Qwen2.5-VL-7B-Instruct":
		messages = [
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": []}
		]

		# Few-shot bloc
		few_shot_bloc = []
		for i_video, (fs_video, fs_prompt) in enumerate(zip(fs_frames, specific_few_shot_prompts)):
			if prompt_strategy['video_placement'] == "above":
				few_shot_bloc += [
						{"type": "video", "video": f"example_{i_video + 1}.mp4", "nframes": num_frames},
						{"type": "text", "text": fs_prompt}
					]
			elif prompt_strategy['video_placement'] == "below":
				few_shot_bloc += [
					{"type": "text", "text": fs_prompt},
					{"type": "video", "video": f"example_{i_video + 1}.mp4", "nframes": num_frames}
				]
			else:
				raise Exception(f"video_placement {prompt_strategy['video_placement']} is not supported.")

		# Now we can build the full prompt depending on prompt_strategy['examples_order']
		if prompt_strategy['examples_order'] == "examples_last":
			messages[1]["content"] += [{"type": "text", "text": prompt}]
			messages[1]["content"] += [{"type": "video", "video": "dummy_video.mp4", "nframes": num_frames}] # core video
			messages[1]["content"] += [{"type": "text", "text": global_few_shot_prompt}]
			messages[1]["content"] += few_shot_bloc
		elif prompt_strategy['examples_order'] == "examples_first":
			messages[1]["content"] += [{"type": "text", "text": prompt}]
			messages[1]["content"] += [{"type": "text", "text": global_few_shot_prompt}]
			messages[1]["content"] += few_shot_bloc
			messages[1]["content"] += [{"type": "text", "text": global_core_prompt}]
			messages[1]["content"] += [{"type": "video", "video": "dummy_video.mp4", "nframes": num_frames}]  # core video
		else:
			raise Exception(f"examples_order {prompt_strategy['examples_order']} is not supported.")

		# Now we can apply the chat template
		text_prompt = tokenizer.apply_chat_template(messages,
		                                            tokenize=False,
		                                            add_generation_prompt=True)
		return text_prompt

	elif model_path == "DAMO-NLP-SG/VideoLLaMA3-7B":
		messages = [
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": []}
		]

		# Few-shot bloc
		few_shot_bloc = []
		for i_video, (fs_video, fs_prompt) in enumerate(zip(fs_frames, specific_few_shot_prompts)):
			if prompt_strategy['video_placement'] == "above":
				few_shot_bloc += [{"type": "image", "image": frame} for frame in fs_video] + [{"type": "text", "text": fs_prompt}]
			elif prompt_strategy['video_placement'] == "below":
				few_shot_bloc += [{"type": "text", "text": fs_prompt}] + [{"type": "image", "image": frame} for frame in fs_video]
			else:
				raise Exception(f"video_placement {prompt_strategy['video_placement']} is not supported.")

		# Now we can build the full prompt depending on prompt_strategy['examples_order']
		if prompt_strategy['examples_order'] == "examples_last":
			messages[1]["content"] += [{"type": "image", "image": frame} for frame in core_frames]
			messages[1]["content"] += [{"type": "text", "text": prompt}]
			messages[1]["content"] += [{"type": "text", "text": global_few_shot_prompt}]
			messages[1]["content"] += few_shot_bloc
		elif prompt_strategy['examples_order'] == "examples_first":
			messages[1]["content"] += [{"type": "text", "text": prompt}]
			messages[1]["content"] += [{"type": "text", "text": global_few_shot_prompt}]
			messages[1]["content"] += few_shot_bloc
			messages[1]["content"] += [{"type": "image", "image": frame} for frame in core_frames]
			messages[1]["content"] += [{"type": "text", "text": global_core_prompt}]
		else:
			raise Exception(f"examples_order {prompt_strategy['examples_order']} is not supported.")

		# Now we can apply the chat template
		return messages

	elif model_path in ["OpenGVLab/InternVideo2_5_Chat_8B", "OpenGVLab/InternVL3-8B"]:
		core_video_prefix = "".join([f"Frame{i + 1} of video 1 : <image>\n" for i in range(len(core_frames))])
		# Few-shot bloc
		few_shot_bloc = []
		for i_example, (fs_video, fs_prompt) in enumerate(zip(fs_frames, specific_few_shot_prompts)):
			video_prefix_fs = "".join([f"Frame{i + 1} of video {i_example + 2}: <image>\n" for i in range(len(fs_frames[i_example]))])
			if prompt_strategy['video_placement'] == "above":
				few_shot_bloc.append(video_prefix_fs + fs_prompt)
			elif prompt_strategy['video_placement'] == "below":
				few_shot_bloc.append(fs_prompt + video_prefix_fs)
			else:
				raise Exception(f"video_placement {prompt_strategy['video_placement']} is not supported.")

		total_prompt = ""
		if prompt_strategy['examples_order'] == "examples_last":
			total_prompt += core_video_prefix
			total_prompt += prompt
			total_prompt += global_few_shot_prompt
			total_prompt += "".join(few_shot_bloc)
		elif prompt_strategy['examples_order'] == "examples_first":
			total_prompt += prompt
			total_prompt += global_few_shot_prompt
			total_prompt += "".join(few_shot_bloc)
			total_prompt += core_video_prefix
			total_prompt += global_core_prompt
		else:
			raise Exception(f"examples_order {prompt_strategy['examples_order']} is not supported.")
		return total_prompt
