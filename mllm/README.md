# Multimodal Large Language Models (MLLMs)

## Installation

### Vad-R1 installation

This repository is based on [Vad-R1](https://github.com/wbfwonderful/Vad-R1).
So, the following installation steps are those mentioned in their repository ;
if you have already created their environment, you can directly 
skip to [additional requirements](#additional-requirements).

```
git clone https://github.com/wbfwonderful/Vad-R1.git
cd Vad-R1

# build environment
conda create -n vad-r1 python=3.11 
conda activate vad-r1
bash setup.sh

# qwen video extraction setting, e.g., max frames, resolutions
# Use the [decord] feature to improve speed
cd src/qwen-vl-utils
pip install -e .[decord]
cd ..
```

Then install the provided version of [transformers](./transformers-main.zip)

```
unzip transformers-main.zip
cd ./transformers-main
pip install .
```

### Additional requirements

Since we have extended [Vad-R1](https://github.com/wbfwonderful/Vad-R1) code, additional packages are required.
You can install them in your activated environment using:
```
pip install -r additional_requirements.txt
```

### Constants

Update in [`constants.py`](constants.py) the variables such as `PROJECT_DIR` that are constant across the repository.

In particular, you may also want to modify the way cluster is detected with the variable `ON_CLUSTER`.

Since Qwen2.5-VL inference implementation relies on `vllm`, that automatically manages GPU,
it is important to set the environment variable `"CUDA_VISIBLE_DEVICES"` from the start
(doing it later may not work as expected). It can be managed in scripts ;
currently, for non-slurm clusters such as Octopus, this is managed by `DEFAULT_GPU`
in [`constants.py`](constants.py).

## Models

Supported models are the following:
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [InternVideo2.5](https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B)
- [InternVL3](https://huggingface.co/OpenGVLab/InternVL3-8B)
- [VideoLLaMA 3](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B)
- [LLaVA-Video](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2) (for zero-shot only)

In practice:
- batch-inference is implemented for Qwen2.5-VL only, so it is the quickest model to run,
both in zero-shot and few-shot
- empirically, InternVL3 is the model that achieves the best results
- depending on the GPU, not all models can run. Qwen2.5-VL and InternVideo2.5 can run on V100,
but other models require bigger GPU's.

Practical remarks if you wish to extend the pool of supported models:
- For `torch` to be compatible with `cuda`, two things are necessary :
  - `torch.version.cuda` (= the CUDA toolkit version used to build that torch) must be 
  <= the CUDA version reported by `nvidia-smi` (= the maximum CUDA version your nvidia driver supports)
  - not only the version, but also the supported architectures matter : `torch.cuda.get_device_capability(device)` 
  must belong to the architectures supported by torch, namely `torch.cuda.get_arch_list()`.
  (For instance, the couple (7,0) returned by the former correspond to the architecture 'sm_70'according to the latter.)
- be careful to the `dtype` when loading a model : `torch.float32` is obviously the best, but is heavier. 
Among 16-bytes encodings, `torch.bfloat16` is more stable than `torch.float16`.
However, `torch.bfloat16` is only supported on GPUs with compute capability of at least 8.0.
- FlashAttention only supports Ampere GPUs or newer.

## Code structure

- [`data/`](data/) : the datasets stored as .csv files
- [`data_analysis/`](data_analysis/) : code for analyzing various aspects of the experiments.
In particular [`inference_analysis.py`](data_analysis/inference_analysis.py) shows how to print qualitative examples of models outputs,
stored as .png pictures representing both frames of the test video and the outputs from the models.
- [`dataset/`](dataset/) : our Dataset class and its utils
- [`dataset_creation/`](dataset_creation/) : the scripts for creating various datasets ;
in particular the demonstration pools.
- [`extract_xclip_features/`](extract_xclip_features/) : code for extracting XCLIP features
- [`getting_result_tables/`](getting_result_tables/) : code for generating the result tables that contain 
performance metrics as well as other useful metrics (such as the proportion of predicted positive)
- [`inference/`](inference/) : code and scripts for generating zero-shot and few-shot inference
- [`inference_results/`](inference_results/) : inference results, stored as .csv in a similar fashion to the initial 
datasets, with the additional columns 'test_pred', 'original_pred' (indicating whether it initially was a None prediction),
'output_text' and, in case of few-shot, the description of few-shots ('few_shot_label_strs' and 'few_shot_guids')
- [`prompts/`](prompts/) : prompts saved in the format `'{exp_name}_{spec}.txt'`.
Formally, a prompt is not attached to any specific task, but they are expected to formulate well the specified task.
- [`statistical_testing/`](statistical_testing/) : contains the code for producing (adjusted) confidence intervals.
- [`constants.py`](constants.py) : contains variables that are constant across the project
- [`global_utils.py`](global_utils.py) : useful functions across the project

## Inference

Both zero-shot and few-shot inference are designed to perform inference on a complete dataset:
to run smaller experiments, simply create a smaller dataset beforehand ; 
you can also restrict the `num_folds` to perform the inference on.

Results are saved in [`inference_results/`](inference_results/) 
as .csv files (in a similar fashion to the base datasets).
They contain the columns 'test_pred', 'original_pred' (indicating whether it initially was a None prediction)
and 'output_text' the full

### Zero-shot inference

Zero-shot inference is implemented in [zero_shot_inf()](inference/zero_shot_inference.py), and it is fully documented.
You can find examples in [zero_shot_experiments/](inference/zero_shot_experiments/), for instance:
```python
zero_shot_inf( 
    # TEST DATASET
    csv_path=f"{DATA_DIR}/mobygaze_dataframe_soft_visual_corrected.csv", # path of the dataset to make inference on
    modality="visual", # for computing modality-specific gold labels
    test_annotator="annotator_2", # annotator to make inference on
    num_folds=[0,1,2,3,4], # test folds to make inference on
    folds_info=FOLDS_INFO, # composition of the folds
    # SAMPLING PARAMETERS
    num_frames=8, # number of frames to sample (the model is not provided an actual video)
    max_pixels=64 * 28 * 28, # max number of pixels for compression
    align_factor=28, # required common divisor to compressed height and width
    # MODEL PARAMETERS
    model_path= "OpenGVLab/InternVideo2_5_Chat_8B", # name of Hugging Face model
    # INFERENCE PARAMETERS
    prompt_characteristics={'exp_name': 'EN_HNS', 'spec': 'new'}, # to identify the .txt prompt file
    inference_batch_size=1, # batch size
    temperature=0.1, # temperature used in sampling during output generation
    top_p=0.9, # for output sampling, consider only the top tokens that together make up top_p probability
    max_tokens=1024, # max number of generated tokens
    do_sample=True, # output is generated based on sampling
    default_prediction=0, # default prediction in case of unsupported answer
    # OTHERS
    spec_str="_zero_shot", # identifier of the experiment
    seed=42, # random seed for reproducibility
)
```

### Few-shot inference

Few-shot inference is implemented in [few_shot_inf()](inference/few_shot_inference.py), and it is fully documented.
You can find examples in [few_shot_experiments/](inference/few_shot_experiments/).

In particular, the choice of few-shot demonstrations is characterized by the initial demonstration pool 
and the selection strategy from this demonstration pool.
Both of those aspects are handled by our code and extensively described in the docstring.

Moreover, because there is more flexibility in the design of a few-shot prompt, we specify the desired design in a
`prompt_strategy` dictionary. For instance:
```python
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
```
From there, you can proceed to few-shot inference, for instance:
```python
few_shot_inf(
    # TEST DATASET
    csv_path=f"{DATA_DIR}/mobygaze_dataframe_soft_visual_corrected.csv",
    modality="visual",
    test_annotator="annotator_2",
    num_folds=[0,1,2,3,4],
    folds_info=FOLDS_INFO,
    # FEW-SHOT
    pooling_type='base', # type of demonstration pool : 'base' = base dataset (with initial annotations)
    few_shot_annotator='annotator_1', # the annotator to select few-shot examples from
    selection_type='balanced_similarity', # examples with largest similarity to the test sample are selected, under the condition the example classes are balanced
    prompt_strategy=prompt_strategy,
    few_shot_num=4, # number of few-shot samples
    few_shot_order='random', # order of presentation of few-shot samples
    feature_extractor="xclip", # the features to compute the cosine similarity from
    # SAMPLING PARAMETERS
    num_frames=8,
    max_pixels=64 * 28 * 28,
    align_factor=28,
    # MODEL PARAMETERS
    model_path= "OpenGVLab/InternVideo2_5_Chat_8B",
    # INFERENCE PARAMETERS
    prompt_characteristics={'exp_name': 'EN_HNS', 'spec': 'new'},
    inference_batch_size=1,
    temperature=0.1,
    top_p=0.9,
    max_tokens=1024,
    do_sample=True,
    default_prediction=0,
    # OTHERS
    spec_str="_few_shot",
    seed=42,
)
```

## Results analysis

Various scripts for analyzing results can be found across the project.
They all share the same design, namely: in order to retrieve the .csv files containing 
inference results (one file for each fold),
an experiment is characterized by a dictionary containing its characteristics, named in a similar fashion
to the actual inference function [zero_shot_inf()](inference/zero_shot_inference.py) or [few_shot_inf()](inference/few_shot_inference.py).

For instance, for the few-shot experiment above:
```python
shot_str = get_shot_str(zero_shot=False, pooling_type='base', few_shot_annotator='annotator_1',
                        selection_type='balanced_similarity', few_shot_num=4, few_shot_order='random')
# a string representing the few-shot strategy (=demo pool + selection strategy)

exp_dict = {'model_path': "OpenGVLab/InternVideo2_5_Chat_8B",
            'prompt_characteristics': {'exp_name': 'EN_HNS', 'spec': 'new'},
            'shot_str': shot_str,
            'exp_name': 'EN_HNS',
            'dataset_name': 'mobygaze_dataframe_soft_visual_corrected',
            'modality': 'visual',
            'seed': 42,
            'spec_str':'_few_shot'}
```

### Quantitative analysis

The function [`main_performances()`](getting_result_tables/get_results_tables.py)
enables to generate a table of results (without confidence interval) for several models 
(and test annotators) at once.
(Simultaneously, it also generates and saves accuracy barplots).
For instance, the following code generates a table containing, 
for each model and test annotator, its test performances averaged across folds and seeds:
```python
all_experiments = {"seed1/model_1": model_dict1, "seed2/model_1": model_dict2, 
                   "seed1/model_2": model_dict3, "seed2/model_2": model_dict4,}
# N.B. : it is recommended to have an actual model name at the end of the keys of all_experiments
all_performances_df = main_performances(
  all_experiments, 
  test_annotators=['annotator_2', 'annotator_1']
)
```

### Qualitative analysis

The function [`qualitative_plots()`](data_analysis/inference_analysis.py) generates plots
with the models outputs for a given video clip.

The function [`qualitative_plots()`](data_analysis/inference_analysis.py) is fully documented.
For instance, in order to generate the plots  for 30 random clips from the full dataset:
```python
qualitative_plots(all_experiments,  test_annotator='annotator_2', num_examples=30,
                  spec_folder='zero_shot_EN_HNS_prompt_1' # the directory of plots/ to save the plots in
                  )
```

## Replication

You can reproduce the experiments from the article using the scripts available in [`inference/`](inference/).
Without any need for you to run them again, inference results are saved in [`inference_results/`](inference_results/),
and [`getting_result_tables/`](getting_result_tables/) and [`statistical_testing/`](statistical_testing/) contain 
the code to reproduce the tables from the article based on them.