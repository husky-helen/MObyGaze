# XCLIP+Transf experiments on MObyGaze

We train models in a two-stage fashion : first training a "representation model"
(that may simply be the representation component of a standard classification model),
then optionnally training a "classification head" on top of those representations.
We test various strategies, both for representation and for classification.

## Installation

You need to install the required packages. The recommended way is :

```
conda create -n mobygazevit python=3.10
conda activate mobygazevit
pip install -r requirements.txt
```

Update in [`constants.py`](constants.py) the variables such as `PROJECT_DIR` that are constant across the repository.

In particular, you may also want to modify the way cluster is detected with the variable `ON_CLUSTER`.
(This variable is used to automatically adapt paths in [`constants.py`](constants.py).)

Note that `tikzplotlib`, which is used in `plot_task_barplot_accuracy()`, is no longer maintained ;
in particular it is not supported in its current state by the most recent versions of `matplotlib`
(while those most recent versions of `matplotlib` are necessary to avoid a down-versioning cascade).
So, in order to use `tikzplotlib` in the code, manual correction of the package is necessary.
Following debugging errors should be enough: [this Stackoverflow answer](https://stackoverflow.com/a/79027395)
describes the main workarounds, while [this second post](https://stackoverflow.com/a/78869532) shows
how to correct the error raised in `_color.py`.

In order to easily identify clips (originally, the 'clip_index' is specific to a movie and an annotator),
first run `add_guid.py`, that will add a 'guid' (identifier column) to the original dataset.
It is important to run it before anything else, so that the guid is consistent across all subsequent datasets.


## Code structure

- [`data_augmentation/`](data_augmentation/) : code for performing different types of data augmentation. 
Resulting datasets (as well as the original dataset) are saved in [`datasets/`](datasets/).
  - [`add_guid.py`](data_augmentation/add_guid.py): adds a 'guid' (identifier column) to the original dataset
  - [`data_restriction.py`](data_augmentation/data_restriction.py) : HNS that do not contain visual queues are dropped
  - [`data_soft_restriction.py`](data_augmentation/data_soft_restriction.py) : HNS that do not contain visual queues are made EN
- [`dataset/`](dataset/): contains the utils for our Dataset objects.
`FeatureDF` is the standard dataset class while `MILDataset` is used for MIL models.
- [`datasets/`](datasets/) : contains the actual .csv datasets.
- [`models/`](models/) : model architectures and utils
- [`results_two_stage/`](results_two_stage/) : test inference results, stored as .csv datasets
- [`two_stage/`](two_stage/) : the core module implementing two-stage inference and subsequent analysis
  - [`audio/`](two_stage/audio/), [`text/`](two_stage/text/), 
  [`visual/`](two_stage/visual/), [`multimodal/`](two_stage/multimodal/) : contains scripts of training and inference 
  based on [`two_stage_train.py`](two_stage/two_stage_train.py)
  - [`analysis/`](two_stage/analysis/) : code for analyzing test results from our models (performance, visualization...)
    - [`significance_and_ci/`](two_stage/analysis/significance_and_ci/) : code for confidence intervals (standard and ANOVA-based)
    - [`analyze_two_stage_disaggregated_results.py`](two_stage/analysis/analyze_two_stage_disaggregated_results.py) :
    generates performance tables (without confidence interval) ; several util functions
    - [`analyze_similarities.py`](two_stage/analysis/analyze_similarities.py) : analyzes the cosine similarities 
    between the feature embeddings of a test and train set : distribution plots and value of 'sigma'
    - [`get_similarity_sigmas.py`](two_stage/analysis/get_similarity_sigmas.py) : from the cosine similarities 
    between the feature embeddings of a test and train set, computes and saves the 'sigma' values
    - [`test_representations.py`](two_stage/analysis/test_representations.py) : some linear and 
    non-linear probers for testing the quality of a "representation"
    - [`validation_barplot_accuracy.py`](two_stage/analysis/validation_barplot_accuracy.py) : accuracy barplots on 
    test and validation sets, as in DMLR paper
    - [`visualize_representations.py`](two_stage/analysis/visualize_representations.py) : generates t-SNE 
    plots of model representations ; additionally generates clustering-based homogeneity metrics
  - [`two_stage_train.py`](two_stage/two_stage_train.py) : trains a model and saves its test inference.
- [`util_data/`](util_data/) : contains the util file [`id_to_title.json`](util_data/id_to_title.json) mapping 
movie ids to their title
- [`constants.py`](constants.py) : contains variables that are constant across the project
- [`get_id_titles.py`](get_id_titles.py) : the script to generate [`id_to_title.json`](util_data/id_to_title.json)
- [`global_utils.py`](global_utils.py) : useful functions, including in particular inference functions


Torch models are saved in `[CHECKPOINTS_DIR]` which should be gitignored.

## Running

### General recommendations

Before running any code, you may need to add the project directory to `PYTHONPATH`,
for instance by adding the following line to your `~/.bashrc`:
```
export PYTHONPATH="${PYTHONPATH}:/[project_path]/vit_mobygaze_copy"
```
Existing `.sh` scripts can be run using :
```
source [script].sh
```

For visualizing tensorboard logs, use the following command in your Linux command line :
```commandline
tensorboard --logdir [logging_directory]
```
Then access the logs at the provided url, for instance by copying it into your browser 
or just clicking on it.

### Training and inference

For training a model and running test inference on it, use the function [`two_stage_train()`](two_stage/two_stage_train.py).
It is fully documented ; for concrete examples, you can see training scripts in [`two_stage/`](two_stage/).
Test inference is automatically performed after training. By default, the trained models are saved 
in `[CHECKPOINTS_DIR]` ; when training a model with the same parameters, the code first checks whether checkpoints 
already exist to avoid useless re-training. Any meaningful parameter change should cause a change in the 
checkpoint path: if this is not the case, you can specify `spec_str` to characterize the model.

Here is a detailed example :

```python
two_stage_train(
  modality="visual", # inputs = visual embeddings
  gt_csv=f"{DATASETS_DIR}/mobygaze_dataframe_soft_visual_corrected.csv", # path of the dataset
  exp_name="EN_HNS", # name of task
  # We train a model for each seed, fold and train annotator :
  folds_info=FOLDS_INFO, # fold composition : default = leave-4-movies-out cross-validation
  random_seeds=[42,43,44,45,46], # different models are created and tested for each random seed
  train_annotators=['annotator_2', 'annotator_1'], # for a given seed and fold, we train one model on Lucile's data and one model on Helen's data
  test_annotators=['annotator_2', 'annotator_1', 'UNION'], # we will produce test inference, for each model, on each of those annotators
  # Thanks to checkpointing, test can easily be performed in a subsequent script
  gpu_index=3, # we run the model on GPU 3
  # TYPE OF REPRESENTATION TRAINING
  representation_architecture="vit", # XCLIP+Transf model architecture
  pool="cls-pooling", # CLS-pooling in the Transformers architecture
  representation_loss="CE", # we use cross-entropy loss
  # TYPE OF CLASSIFICATION TRAINING
  consider_classif=False, # we train no additional classification head
  # OTHERS
  balancing=True, # the train set is balanced, using oversampling of the minority class
  save_model=True, # models checkpoints are saved
  batch_size=128, # the "representation model" is trained with batch size 128
  use_trained_representation_models=True, # if a checkpoint already exists for the "representaiton model", we use it
  spec_str="_DMLR", # experiment identifier
)
```

Note that, for objectification tasks, training and test classes are generated based on the 'label' column.
So, typically, CMSD datasets should be created beforehand. In the repository, they correspond to the
`mobygaze_dataframe_soft_[modality]_corrected.csv`datasets.

### Analysis of results

Various scripts for analyzing results can be found in [`two_stage/analysis/`](two_stage/analysis/)
and are listed in [the code structure](#code-structure) with a brief description.
They all share the same design, namely: in order to retrieve the .csv files containing 
inference results (one file for each fold),
a model is characterized by a dictionary containing its characteristics, named in a similar fashion
to the actual inference function [`two_stage_train()`](two_stage/two_stage_train.py).

For instance:
```python
exp_dict = {'exp_name': 'EN_HNS',
            'train_annotator': 'annotator_2',
            'seed': 42,
            # TYPE OF REPRESENTATION TRAINING
            'representation_architecture': 'vit',
            'representation_loss': 'CE',
            'batch_size': 128,
            'full_model': True, # means we do not append it any classification head
            # For other architectures and training strategies, we may specify additional arguments
            # FOR DETAILED DIRECTORY
            'modality': 'visual',
            'pool': 'cls-pooling',
            'spec_str': '_DMLR',
            'dataset_name': 'mobygaze_dataframe_soft_visual_corrected',
            }
```
For a given model, the dataframe of all predictions (ie. over all folds) can be retrived using 
[`get_prediction_df()`](two_stage/analysis/analyze_two_stage_disaggregated_results.py).
It returns a dataframe similar to the initial dataset, with the additional columns 
'test_proba' and 'predictions' (an integer column corresponding to the classification task).
For instance:
```python
predictions_df = get_prediction_df(
  **exp_dict,
  test_annotator='annotator_1',
  inference_set='test'
)
# inference_set needs to be specified, as train and val inference can optionnally be saved as well
```
And, for a given model, [`get_task_result()`](two_stage/analysis/analyze_two_stage_disaggregated_results.py)
generates performance results, without confidence intervals. For instance, the following code generates a table 
containing, for each fold, its test performance regarding various performance metrics (AUC scores, F1, etc.):
```python
performance_df = get_task_result(
  exp_dict,
  model_name="vit", # the name that will appear in the 'model' column
  task="EN_HNS", # the task according to which true labels will be computed
  # For test results to make sense, task should be the same as the one the model was trained for,
  # but it may be interesting, for instance, to check if a EN_HNS model actually works as a 'Body' detector
  per_movie=False, # whether we generate results differentiated by movie
  per_label=False, # whether we generate results differentiated by true label
  detailed=True, # whether we average the resulting table
  test_annotator='annotator_1',
  inference_set='test',
)
```
The function [`main_performances()`](two_stage/analysis/analyze_two_stage_disaggregated_results.py)
enables to generate a table of results (without confidence interval) for several models 
(and test annotators) at once.
(Simultaneously, it also generates and saves plots: accuracy barplots and 
accuracy/heterogeneity trade-off maps). For instance, the following code generates a table containing, 
for each model and test annotator, its test performances averaged across folds and seeds:
```python
all_experiments = {"seed1/model_1": model_dict1, "seed2/model_1": model_dict2, 
                   "seed1/model_2": model_dict3, "seed2/model_2": model_dict4,}
# N.B. : it is recommended to have actual model_name at the end of the all_experiments keys
all_performances_df = main_performances(
  all_experiments, 
  test_annotators=['annotator_2', 'annotator_1']
)
```

### Confidence intervals

[`standard_ci()`](two_stage/analysis/significance_and_ci/significance_analysis.py) generates a table with standard confidence intervals, for instance:
```python
standard_ci(
  all_experiments,
  groupby_variable="exp_name", # the group-by variable enables to compare other things than models
  alpha=0.05, # for (1-alpha)-confidence intervals
  metrics=["positive_pr_auc", "negative_pr_auc", "auc_pr_macro", "accuracy"], # the metrics of interest
  spec_str=f'_results_by_exp_name', # identifier of the resulting table
)
```

[`significance_analysis()`](two_stage/analysis/significance_and_ci/significance_analysis.py) analyzes the impact of a given factor (typically the model architecture) 
on performance. In particular, it runs statistical tests checking the hypotheses underlying confidence intervals 
(such as homoscedasticity) and generates the tables with the ANOVA-based confidence intervals.
For instance:
```python
significance_analysis(
  all_experiments, 
  factor="model",
  alpha=0.05, 
  metrics=["positive_pr_auc", "negative_pr_auc", "auc_pr_macro"],
  spec_str=f'_EN_HNS_vit_vs_random_vs_mlp' # identifier of the resulting table
)
```

For more examples, the folder [`significance_and_ci/`](two_stage/analysis/significance_and_ci/)
contains the scripts for generating DMLR tables.

## Replication

You can reproduce the models from the article and their training using the scripts available in [`two_stage/`](two_stage/).
Without any need for you to retrain the models, inference results are saved in [`results_two_stage/`](results_two_stage/),
and [`significance_and_ci/`](two_stage/analysis/significance_and_ci/) contains the code to reproduce the tables from the article based on them.
