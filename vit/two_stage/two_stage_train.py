import json

import numpy as np
import torch
import torchvision

#print(torch.__version__)
#print(torchvision.__version__)

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau  # for the scheduler
from torch.utils.tensorboard import SummaryWriter
from random import shuffle
from functools import partial
from sklearn.metrics import f1_score
from typing import List
import os
from scipy.special import softmax
from math import ceil

from dataset.dataset import FeatureDF
from dataset.mil_dataset import MILDataset
from models.pcgrad import PCGrad
from global_utils import *
from models.losses_SupContrast import IntraClusterSupConLoss, GlobalSupConLoss
from models.mil_loss import ranking
from models.differential_sampler import DifferentialSampler, FullMovieDifferentialSampler, TemporalDifferentialSampler
from two_stage.analysis.test_representations import test_representations_various_probers, get_linear_clf
from models.scheduling_functions import exponential_decay
from constants import *

import warnings
from sklearn.exceptions import UndefinedMetricWarning
#warnings.filterwarnings("error")
warnings.filterwarnings(
    action='error', message='',
    category=UndefinedMetricWarning
)


# From cartography.cartography.selection.selection_utils in DataMaps code
# The advantage of using their way of saving is to also use their plot code etc.
def log_training_dynamics(logging_dir: os.path,
						  epoch: int,
						  train_ids: List[int],
						  train_logits: List[List[float]],
						  train_golds: List[int]):
	"""
	Saves training dynamics (logits) from given epoch as records of a `.jsonl` file.
	"""
	# The arguments also can be np arrays, except train_logits that would lead to an exception (because it's 2D)
	td_df = pd.DataFrame({"guid": train_ids,
						  f"logits_epoch_{epoch}": train_logits,
						  "gold": train_golds})
	# Create directory for logging training dynamics, if it doesn't already exist.
	if not os.path.exists(logging_dir):
		os.makedirs(logging_dir)
	epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
	td_df.to_json(epoch_file_name, lines=True, orient="records")


def get_repr_strs(representation_architecture, representation_loss,
				 global_contrastive, contrastive_type, weighted_contrastive, temperature,
				 nb_movies_per_batch, nb_clips_per_movie,
				 nb_segment, length_segment,
				 batch_size, train_annotator_backbone, representation_backbone=None,
				  target_model=None,
				  train_on_target=False, pseudo_labeling=False, confidence_based_pseudo_labeling=False, budget=None,
				  fraction_train=None,
				  randint_train=True,
				  **kwargs
				 ):
	"""
	Generates a string that characterizes the architecture specified by the arguments.
	The meaning of arguments is the same as in the function two_stage_train().

	"""

	if "random" in representation_architecture:
		return "random", "random"

	type_representation = f"{representation_architecture}_{representation_loss}"
	if "MIL" in representation_loss and not ("shot_based_segments" in representation_loss):
		length_segment_str = f"_{length_segment}_length" if "fixed_length" in type_representation else ""
		type_representation += f"_{nb_segment}_segments" + length_segment_str
	elif "target_embedding" in representation_loss:
		target_repr_str, target_type_representation = get_repr_strs(**target_model)
		type_representation += "_" + target_repr_str

	contrastive = representation_loss == "contrastive"
	if not contrastive:
		if representation_architecture == "xclipemb":
			repr_str = type_representation
		elif "differential" in representation_architecture:
			repr_str = f"{type_representation}_bs_{nb_movies_per_batch}x{nb_clips_per_movie}"
		else:
			repr_str = f"{type_representation}_bs_{batch_size}"

	else:
		global_contrastive_str = "global_contrastive" if global_contrastive else "intra_contrastive"
		weighted_contrastive_str = "weighted_" if (weighted_contrastive and not global_contrastive) else ""
		repr_str = f"{weighted_contrastive_str}{global_contrastive_str}_{contrastive_type}_temp_{temperature}_bs_{batch_size}"

	if representation_architecture == "frozen_context_differential_vit":
		frozen_repr_str, frozen_type_representation = get_repr_strs(**representation_backbone)
		additional_str = f"_{train_annotator_backbone}_{frozen_repr_str}_backbone"
		type_representation += additional_str
		repr_str += additional_str

	if "cem" in representation_architecture and randint_train:
		repr_str += "_randint_train"

	if pseudo_labeling:
		if confidence_based_pseudo_labeling:
			repr_str += "_confidence_based_pl"
		repr_str += f"_budget_{budget}"

	if train_on_target:
		repr_str += "_train_on_target"

	if fraction_train is not None:
		repr_str += f"_fraction_train_{round(fraction_train, 2)}"

	return repr_str, type_representation


def two_stage_train(modality, gt_csv, random_seeds, train_annotators, test_annotators,
					num_folds, folds_info, pool, spec_str,
					exp_name,
					gpu_index,
					# TYPE OF REPRESENTATION TRAINING
					representation_architecture, representation_loss,
					# CONTRASTIVE ARGUMENTS
					global_contrastive=None, contrastive_type=None, weighted_contrastive=None, temperature=None,
					# TYPE OF CLASSIFICATION TRAINING
					consider_classif=False, type_classif='mlp', similarity_betas=[], similarity_clustering_ks=[],
					# FOR DIFFERENTIAL (REPRESENTATION) TRAINING
					nb_movies_per_batch=None, nb_clips_per_movie=None,
					# FOR MIL
					nb_segment=16, length_segment=5,
					# FOR TARGET_EMBEDDING
					target_model=None,
					# FOR APOSTERIORI AND FROZEN_CONTEXT AND BACKBONE CEM
					representation_backbone=None, train_annotator_backbone=None,
					# FOR PSEUDO-LABELING TRAINING
					pseudo_labeling=False, confidence_based_pseudo_labeling=False,
					pseudo_labeling_val_to_unlabeled=False,
					also_relabel_val_set=None,
					max_pseudo_label_proportion=1.,
					stepping_threshold=0.2,
					budget=10000,
					# FOR CEM
					concept_loss_weight=1.,
					randint_train=True,
					# OTHERS
					balancing=True, save_model=True, batch_size=128, batch_size_class=128,
					use_trained_representation_models=True,
					use_trained_classification_models=True,
					evaluate_representation=False,
					evaluate_representation_classifier=True,
					test_on_custom_dataset=None,
					custom_annotator='UNION',
					no_cheating_val=True,
					no_cheating_test=True,
					variable_threshold=False,
					save_val_inference=False,
					save_train_inference=False,
					save_target_train_losses=False,
					untrained=False,
					custom_clips=False,
					save_training_dynamics=False,
					test_unlabeled_set=False,
					train_on_target=False,
					fraction_train=None,
					additional_train_guids=None,
					print_nb_parameters=False,
					specified_visual_feature_dir=None
					):
	"""
	Trains a model, categorized into a 'representation component' that can learn (possibly in a hidden state)
	a representation, and an optional classification head. Saves this model and its test inference results.

	:param modality: whether the modality of interest is "visual", "text" or "audio"
	or any sequence of modalities separated by '+' (for instance 'text+visual')
	:param gt_csv: the .csv path of the dataset
	:param random_seeds: the random seeds to perform the experiments (as man experimentsare run as the number of seeds)
	The random operations that are thus made reproducible are model initialization and dataset shuffling/sampling.
	:param train_annotators: the annotators used to train a model (as many trainings are performed as number of train_annotators)
	Possible values are 'annotator_2', 'annotator_1', 'UNION' and are self-explanatory
	:param test_annotators: the annotators used to test a model (as many tests are performed as number of test_annotators)
	Possible values are 'annotator_2', 'annotator_1', 'UNION' and are self-explanatory
	:param num_folds: iterable of the fold indices (which are the keys of folds_info)
	:param folds_info: a dictionary describing the composition of folds (regarding imdbs)
	The expected structure is folds_info = {i_fold: {"test": list_of_test_imdbs, "validation": list_of_validation_imdbs}} ;
	see FOLDS_INFO as an example ; train_imdbs are defined as complementary of test and val imdbs
	:param pool: the type of pooling used in Transformers model.
	Possible values are 'cls-pooling' (the one typically used), 'avg-pooling'
	:param spec_str: a string aiming to characterize an experiment
	(from another experiment whose hyperparameters used in naming of chekpoints and results are the same)
	:param exp_name: the name of the classification task. Possible values are '[concept_name]', 'all_visual' (multi-label prediction of visual concepts),
	and all objectification classifications : "EN_HNS", "ENHN_S", "EN_S", "EN_HN_S"
	N.B. : it automatically characterizes a "classif_info" converting concepts and obj labels into labels
	:param gpu_index: the gpu index to run the model on
	:param representation_architecture: the type of representation model
	N.B. : in terms of terminology, it would be more appropriate to call it "first_stage_model", which can be
	a pure representation model as well as a full model
	(that produces a representation as intermediate results/hidden output).
	(N.B. : Note that, as I implemented MLP at the end, representation_architecture="mlp" is not equivalent to
	representation_architecture="xclipemb" and type_classif="mlp" ; see the difference between MLP and ClassificationHead).
	Possible values are :
	"xclipemb", "representation_vit", "vit"
	"differential_vit", "differential_vit_negative", "differential_vit_iterative", "complex_differential_vit",
	"aposteriori_differential_vit", "double_differential_vit", "scheduled_double_differential_vit",
	"frozen_context_differential_vit", "gated_differential_vit", "complex_gated_differential_vit",
	"attentionlike_gated_differential_vit",
	"differential_vit_selective", "differential_vit_selective_negative",
	"multimodal_vit"
	(it is also possible to add "temporal" at the end of the selective models ; it is not supported for others,
	because of strict name equality matching, but it can easily),
	"dann", "dann_binary", "dann_by_prop_[n]", "differential_dann[_type_of_dann]"
	For all models containing "vit" (except multimodal), there are equivalent models containing "mlp" and "bigmlp"
	Plus "cem_vit", "cem_multimodal_vit", "backbone_cem_vit", "backbone_cem_multimodal_vit"
	N.B.:by definition, no random model is trained here. However, in analysis functions, in order to get random results,
	choose representation_architecture='random'

	:param representation_loss: "CE", "fixed_length_MIL", "fixed_nb_segments_MIL", "shot_based_segments_MIL"
	"contrastive", "target_embedding"
	N.B. : uppercase ensure not to be tricked by other name components when performing string matching
	N.B. : "contrastive" and "target_embedding" losses are representation losses

	:param global_contrastive: if representation_loss="contrastive", if the contrastive loss applies globally,
	or at (intra) cluster level
	:param contrastive_type: if representation_loss="contrastive", indicates whether the specific contrastive loss is
	'supcon', 'pushing_away', 'mixed'
	:param weighted_contrastive: if representation_loss="contrastive" and global_contrastive=False,
	indicates if the contrastive loss is weighted by (the inverse of) the cluster size
	:param temperature: temperature parameter in the contrastive loss
	:param consider_classif: whether we want to include classification in the process, ie. evaluation and (if needed) training
	:param type_classif: the type of classification head to use: in ['knn', 'logistic', 'mlp', 'similarity', 'similarity_clustering']
	This is only used if consider_classif=True.
	:param similarity_betas: the beta values to test (by cross-validation) for the similarity-based classifier
	:param similarity_clustering_ks: the k values to test (by cross-validation) for the similarity(-clustering)-based classifier

	:param nb_movies_per_batch: for differential training (requiring many clips from the same movie in the same batch),
	indicates the number of movies present in the batch
	:param nb_clips_per_movie: for differential training (requiring many clips from the same movie in the same batch),
	indicates the number of clips per movie present in the batch

	:param nb_segment: the number of segments in the context of MIL training :
	if "fixed_nb_segments_MIL", it simply corresponds to the number of segments ;
	if "fixed_length_MIL", it is still used as the maximal authorized number of segments
	(if exceeded, the clip is processed similarly to "fixed_nb_segments_MIL")
	:param length_segment: the length of a segment in the context of "fixed_length_MIL" training

	:param target_model: a dict containing the characteristics of the target model (to retrieve it)
	for learning a representation model using "target_embedding" loss

	:param representation_backbone: a dict containing the characteristics of the backbone model (to retrieve it)
	for aposteriori or frozen_context models. It must contain architecture characteristics, and it may contain
	additional training characteristics (exp_name, pool, spec_str, dataset_name ;
	for the backbone to be relevant the modality must be the same anyway)
	:param train_annotator_backbone:

	:param pseudo_labeling: whether we train the model in a pseudo_labeling fashion (more precisely, curriculum labeling)
	:param confidence_based_pseudo_labeling: whether the curriculum labeling is based on confidence.
	Only used if pseudo-labeling=True
	:param pseudo_labeling_val_to_unlabeled: if set to True, unlabeled set = val set (according to folds info) ;
	else, it corresponds to observations indicated as 'labeled'=0 (and belong to train) in the gt_csv
	(Note that, for generalization of the code, if pseudo_labeling_val_to_unlabeled, still, in the code,
	val movies are still eventually converted into train movies, and the appropriate 'labeled' column is created
	:param also_relabel_val_set: if not None, a list of fold_budgets we are supposed to relabel the val set on,
	according to model train on D_L+D_R. (This makes sense when it is the train set that is split into D_L+D_U)
	:param max_pseudo_label_proportion: the pseudo-label proportion in the unlabeled set
	that we want to reach at the end of the training cycles
	:param stepping_threshold: the percentage that we add at every training cycle to the pseudo_label_proportion
	:param budget: the number of observations that we are willing to re-annotate (per fold)

	:param concept_loss_weight: for CEM models, the value of alpha in the total loss formula
	total_loss = task_loss + alpha * concept_loss
	:param randint_train: for CEM models, whether we perform training using RandInt strategy

	:param balancing: whether the train set is balanced
	:param save_model: whether to save the trained models. N.B. : for now it is more or less mandatory not to raise error
	:param batch_size: training batch size for the core (representation) model
	:param batch_size_class: training batch size for the core classification model, if it is MLP and trained in a successive fashion
	:param use_trained_representation_models: if we use already saved representation model (for the same hyperparameters)
	in order not to train it again from scratch.
	If False, it will replace already existing models with the same hyperparameters by newly trained models
	:param use_trained_classification_models: if we use already saved classification model (for the same hyperparameters
	and the same representation head, as the classification head is always dependent on the representation head)
	in order not to train it again from scratch. Note that it is incompatible with simultaneous. Moreover,
	if False, it will replace already existing models with the same hyperparameters by newly trained models
	:param evaluate_representation: whether to include the representation evaluation (during its training,
	and on the final representation model)
	:param evaluate_representation_classifier: whether to include the classification evaluation of the representation model
	(in case the representation model performs both, which is the case for all except "contrastive", "xclipemb")

	:param test_on_custom_dataset: (optional) the path of a .csv dataset (created beforehand) to perform test inference on
	:param custom_annotator: the annotator to test in the optional custom test dataset

	:param no_cheating_val: whether we authorize the access to true test labels during validation
	(Obviously we should not, but cheating may be useful for experimenting)
	:param no_cheating_test: whether we authorize the access to true test labels during test
	(Obviously we should not, but cheating may be useful for experimenting)

	:param variable_threshold: whether, at test time, we also compute the optimal threshold
	(according to the F1-macro on validation set). N.B. : this will not influence stored predictions,
	as we only store output probabilities ; but optimal_threshold will be stored as well and can be used later
	:param save_val_inference: whether, at test time, we also compute and save the inference on the validation set
	:param save_train_inference: whether, at test time, we also compute and save the inference on the train set
	:param save_target_train_losses: whether we save the target train losses, ie., for each fold,
	the training loss corresponding to the best validation loss, ie. to the best model.
	Stored training losses will be used for computing training dynamics (because the former
	:param untrained: does not train the model. Beware it will still be saved and used for inference
	(it is its very motivation). To avoid to overwrite the equivalent trained model,
	spec_str is automatically set to "_untrained"
	:param custom_clips: whether the custom dataset is of the kind "custom_clip",
	ie. clips were created by assembling subclips from various movies
	:param save_training_dynamics: whether training dynamics are saved
	:param test_unlabeled_set: whether to perform testing on the unlabeled set (associated to the training set)
	at the end of training. N.B.: This is default for pseudo-labeling
	:param train_on_target: standard training is performed with early stopping.
	An alternative is training in order to achieve a given target loss typically, when there is no available validation set)
	:param fraction_train: the fraction of the train set we include ; all by default
	:param additional_train_guids: if not None, a list that contains, for each fold, the guids to add in the train set
	(that otherwise should be in test or validation).
	N.B. : Those guids will still be in test or validation, but also in train
	:param print_nb_parameters: whether we print the number of parameters of the core model
	:param specified_visual_feature_dir: in case we want to specify a feature dir different from the default one

	:return:
	"""

	classif_info = classif_info_from_exp_name(exp_name)

	torch.autograd.set_detect_anomaly(True)

	if untrained:
		spec_str = "_untrained"

	if test_unlabeled_set:
		assert ("vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture)

	if train_on_target:
		assert ("vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture)
		assert representation_loss == 'CE'

	# N.B. : for saving training dynamics, we set the whole dataset to 'train'
	if save_training_dynamics:
		assert ("vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture)
		assert representation_loss == 'CE'
		balancing = False
		print(f"\n!!! Beware balancing is set to False for computing training_dynamics\n")
		spec_str = "_training_dynamics" # we make sure not to overwrite a proper model
		folds_info = {0: {"train": ALL_MOVIES, "test": [], "validation": []}}
		num_folds = [0]
		print(f"\n!!! Beware all movies are now considered train movies (no test or val) for computing training_dynamics\n")

	# To handle the difference between binary and multiclass classification
	real_n_classes = classif_info['real_n_classes']
	# because in the models the argument 'num_classes' is actually num_output_neurons
	num_output_neurons = classif_info['num_output_neurons']

	# If it is multimodal, we want to unify "text+visual" and "visual+test", for instance
	modalities = modality.split("+")
	modalities.sort()
	modality = "+".join(modalities)
	# We also compute several useful variables for multimodal, that still make sense for unimodal case
	nb_modalities = len(modalities)
	multimodal = nb_modalities > 1
	if "MIL" in representation_loss:
		assert not multimodal
	
	# To handle the multi-modal case (without losing the generalization of the code)
	my_collate_fn = multimodal_custom_collate_fn_with_guids_movie_ids if multimodal else custom_collate_fn_with_guids_movie_ids

	# Retrieve the dataset name (for precise naming of directories)
	dataset_name = gt_csv.split('/')[-1][:(-4)] + ("" if balancing else "_unbalanced")
	unbalanced_dataset_name = gt_csv.split('/')[-1][:(-4)] + "_unbalanced"
	# In case of pseudo-labeling based experiments, the dataset_name of the ttl directory does not exactly match the current dataset_name :
	if 'labeled' in unbalanced_dataset_name:
		noun_components = unbalanced_dataset_name.split("_")
		labeled_component = ['labeled' in component for component in noun_components]
		labeled_id = labeled_component.index(True)
		unbalanced_dataset_name = "_".join(noun_components[:labeled_id]) + "_unbalanced"
		# This does nothing else but reconstructing the name of the original dataset target_train_losses were computed on
	# regardless od current balancing, the target train losses (and thus their directory) have always been performed with unbalanced training

	# In order to map train film ids to class indices (for DANN), we first need to have a mapping from imdb keys to ids
	# (indeed, unlike ids, we have easy access to the imdb keys in the train set, for a given fold)
	df = pd.read_csv(gt_csv, sep=";", converters={'concepts': pd.eval})
	imdb_to_id = dict()
	for imdb_k in df["imdb_key"].unique():
		imdb_to_id[imdb_k] = int(df.query("imdb_key == @imdb_k")["movie_id"].iloc[0])

	# N.B. : As many arguments of this function are specific to the contrastive training, we could have separated
	# the code of other representation learnings in another function,
	# but this way we guarantee that all other training hyperparameters are identical (and thus their results comparable)

	type_probers = ['svm', 'logistic', 'knn', 'mlp', 'prior', 'uniform'] # for representation testing

	# For saving the models, we need to characterize them by a str
	repr_str, type_representation = get_repr_strs(representation_architecture, representation_loss,
				 global_contrastive, contrastive_type, weighted_contrastive, temperature,
				 nb_movies_per_batch, nb_clips_per_movie,
				 nb_segment, length_segment,
				 batch_size, train_annotator_backbone, representation_backbone, target_model,
												  train_on_target, pseudo_labeling,
												  confidence_based_pseudo_labeling, budget,
												  fraction_train,
												  randint_train
												  )
	# repr_str is more detailed than type_representation, so that checkpoints of similar but different models are not overwritten
	type_classif_str = type_classif if not "mlp" in type_classif else f"{type_classif}_bs_{batch_size_class}"

	# Depending on the modality, some parameters may change
	modality_max_seq_len = {'visual': 160, 'text': 512, 'audio': 160}
	modality_feature_dim = {'visual': 512, 'text': 768, 'audio': 768}
	# For multimodal case, max_seq_len and feature_dim are lists, while for unimodal case they are ints :
	# this should cause no problem, as the nly places where those variables appear are when calling
	# FeatureDF, load_representation_model and MIL ; and we don't run MIL in multimodal setting
	if multimodal:
		feature_dim = [modality_feature_dim[mod] for mod in modalities]
		max_seq_len = [modality_max_seq_len[mod] for mod in modalities]
	else:
		feature_dim = modality_feature_dim[modality]
		max_seq_len = 40 if "MIL" in representation_loss else modality_max_seq_len[modality]
		# 160 is the standard value in the project, but with MIL (no matter which specific type of MIL) resulting segments
		# should be much shorter ; thus padding tokens would consume useless memory

	batch_size = batch_size
	dim = 128  # 768 use bigger working dimension than feature dimension (according to the paper)
	# We keep the hidden "dim" across all architectures : this way MLP models will be much smaller in capacity than ViT,
	# but their classification heads will be similar, and experience showed the results are not worse than with dim=256
	heads = 8
	dim_head = dim // heads  # This ensures dim is divisible by heads
	mlp_dim = 1024  # 1536 #2024 Julie #1536  # 2 times the input dimension (2024 in the previous experiment)
	depth = 2  # 2 #6 original paper
	dropout = 0.1
	emb_dropout = 0.1
	total_epochs = 100
	min_epochs = 1
	total_epochs_class = 100
	min_epochs_class = 1

	for seed in random_seeds:
		print(f"\n\n\n### SEED {seed}")
		setup_seed(seed)

		for train_annotator in train_annotators:
			print(f"\n\n## TRAIN ANNOTATOR {train_annotator}")
			annotator_target_train_losses = []
			# we save, for each fold, the target train loss (ie. the one corresponding to the best validation loss),
			# for each fold (but we don't care about the seed, as the result should be independent of the seed)

			cv_results_all = {type_prober: [] for type_prober in type_probers}
			cv_results_classif = []
			cv_results_classif_core = []  # if the core (representation) model is actually also a classifier

			for fold_idx in num_folds:
				print(f"\n# FOLD_IDX {fold_idx}")

				# Unlike standard training, pseudo-labeling training consists of several training cycles
				training_cycles_finished = False
				training_cycle_id = 0
				pseudo_label_proportion = 0. # the number of new pseudo-labels we include at each training cycle
				pseudo_labeling_weight = 1.
				remaining_budget = budget
				# We keep track of the pseudo-labels we include at each step (that may be identical to original labels)
				pseudo_labels_guids = []
				pseudo_labels = []
				# we keep track of those different from the original labels,
				# that are the most interesting and will be used to save the relabeled dataset
				all_different_pseudo_labels_guids = []
				all_different_pseudo_labels = []

				while not training_cycles_finished:
					if pseudo_labeling:
						print(f"\ntraining_cycle_id {training_cycle_id}...")
						print(f"remaining_budget :{remaining_budget}")
						print(f"len(pseudo_labels) : {len(pseudo_labels)}")

					# For checkpointing models and inference results accurately, we also specify the training_cycle_id,
					# ... except for the last cycle, which corresponds to the "true" model
					# (and thus can be accessed like classical models)
					pseudo_labeling_str = f"_cycle_{training_cycle_id}" if pseudo_labeling else ""

					# DATASETS
					print(f"LOADING DATASETS...")
					train_dataset = FeatureDF(modality=modality, gt_csv=gt_csv, split="train", fold_idx=fold_idx,
											  folds_info=folds_info, classif_info=classif_info, max_seq_len=max_seq_len,
											  seed=seed, annotator=train_annotator, balancing=balancing,
											  with_guid=True, with_id=True,
								  representation_architecture=representation_architecture,
											  pseudo_labeling_xt=pseudo_labeling, pseudo_labels_guids=pseudo_labels_guids, pseudo_labels=pseudo_labels,
											  pseudo_labeling_val_to_unlabeled=pseudo_labeling_val_to_unlabeled,
											  fraction_train=fraction_train,
											  additional_train_guids=additional_train_guids,
											  specified_visual_feature_dir=specified_visual_feature_dir
											  )
					# unbalanced_train_dataset will be used for saving train inference
					unbalanced_train_dataset = FeatureDF(modality=modality, gt_csv=gt_csv, split="train", fold_idx=fold_idx,
											  folds_info=folds_info, classif_info=classif_info, max_seq_len=max_seq_len,
											  seed=seed, annotator=train_annotator, balancing=False,
											  with_guid=True, with_id=True,
											  representation_architecture=representation_architecture,
											  pseudo_labeling_xt=pseudo_labeling,
											  pseudo_labels_guids=pseudo_labels_guids, pseudo_labels=pseudo_labels,
											  pseudo_labeling_val_to_unlabeled=pseudo_labeling_val_to_unlabeled,
											  fraction_train=fraction_train,
														 additional_train_guids=additional_train_guids,
											  specified_visual_feature_dir=specified_visual_feature_dir
											  )
					print(f"len(train_dataset) (balanced={balancing}) : {len(train_dataset)}")
					val_dataset = FeatureDF(modality=modality, gt_csv=gt_csv, split="validation", fold_idx=fold_idx,
											folds_info=folds_info, classif_info=classif_info, max_seq_len=max_seq_len,
											seed=seed, annotator=train_annotator, with_guid=True, with_id=True,
								  representation_architecture=representation_architecture,
											  specified_visual_feature_dir=specified_visual_feature_dir)

					train_loader_class = DataLoader(train_dataset, batch_size=batch_size_class, shuffle=True,
													num_workers=19,
													collate_fn=my_collate_fn, drop_last=False)
					val_loader = DataLoader(val_dataset,
											batch_size=len(val_dataset) if "differential" in representation_architecture else batch_size,
											shuffle=False, num_workers=19, collate_fn=my_collate_fn)

					if pseudo_labeling or test_unlabeled_set:
						unlabeled_dataset = FeatureDF(modality=modality, gt_csv=gt_csv, split="train", fold_idx=fold_idx,
											  folds_info=folds_info, classif_info=classif_info, max_seq_len=max_seq_len,
											  seed=seed, annotator=train_annotator, balancing=False,
											  with_guid=True, with_id=True,
								  representation_architecture=representation_architecture,
													  pseudo_labeling_dul=True,
													  pseudo_labeling_val_to_unlabeled=pseudo_labeling_val_to_unlabeled,
											  specified_visual_feature_dir=specified_visual_feature_dir
											  )
						print(f"len(unlabeled_dataset) : {len(unlabeled_dataset)}")
						pseudo_label_number = ceil(stepping_threshold * len(unlabeled_dataset))
						# the number of new-pseudo-labels at each step ; it is easier to manipulate than stepping threshold,
						# as the number of remaining unlabeled will change at each training cycle
						unlabeled_loader = DataLoader(unlabeled_dataset,
											batch_size=len(unlabeled_dataset) if "differential" in representation_architecture else batch_size,
											shuffle=False, num_workers=19, collate_fn=my_collate_fn)
						# The inference on this dataset will correspond to pseudo-labels

					# Also the datasets and loaders for MIL.
					# N.B. : MIL is well-defined for binary classification only
					if "MIL" in representation_loss:
						mil_train_ndataset = MILDataset(modality=modality, gt_csv=gt_csv, split="train", fold_idx=fold_idx,
														folds_info=folds_info, classif_info=classif_info,
														max_seq_len=max_seq_len,
														seed=seed, annotator=train_annotator, balancing=True,
														with_guid=True, with_id=True,
														is_normal=True, feature_dim=feature_dim, nb_segment=nb_segment,
														length_segment=length_segment, type_representation=type_representation)
						mil_train_adataset = MILDataset(modality=modality, gt_csv=gt_csv, split="train", fold_idx=fold_idx,
														folds_info=folds_info, classif_info=classif_info,
														max_seq_len=max_seq_len, seed=seed, annotator=train_annotator,
														balancing=True, with_guid=True, with_id=True,
														is_normal=False, feature_dim=feature_dim, nb_segment=nb_segment,
														length_segment=length_segment, type_representation=type_representation)
						mil_val_dataset = MILDataset(modality=modality, gt_csv=gt_csv, split="validation", fold_idx=fold_idx,
													 folds_info=folds_info, classif_info=classif_info,
													 max_seq_len=max_seq_len,
													 seed=seed, annotator=train_annotator, with_guid=True, with_id=True,
													 is_normal=True, feature_dim=feature_dim, nb_segment=nb_segment,
													 length_segment=length_segment, type_representation=type_representation)

						mil_val_loader = DataLoader(mil_val_dataset, batch_size=len(mil_val_dataset) if "differential" in representation_architecture else batch_size,
													shuffle=False, num_workers=19, pin_memory=True, collate_fn=my_collate_fn)

					# We define the train loader separately from other loaders,
					# as it is defined in different ways depending on if we use a differential model
					if ("differential" not in representation_architecture) or nb_movies_per_batch is None or nb_clips_per_movie is None:
						# Note that, in differential CE, if nb_movies_per_batch is None or nb_clips_per_movie is None,
						# we get the standard dataloader : this means the number of clips to build a movie context may be very low
						train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=19,
												  collate_fn=my_collate_fn,
												  drop_last=(representation_loss == "contrastive"))
						# unshuffled_train_loader will be used for saving train inference
						unshuffled_train_loader = DataLoader(unbalanced_train_dataset, batch_size=batch_size, shuffle=False, num_workers=19,
												  collate_fn=my_collate_fn,
												  drop_last=(representation_loss == "contrastive"))
						if "MIL" in representation_loss:
							mil_train_nloader = DataLoader(mil_train_ndataset, batch_size=batch_size, shuffle=True, num_workers=19,
														   collate_fn=my_collate_fn,
														   drop_last=(representation_loss == "contrastive"))
							mil_train_aloader = DataLoader(mil_train_adataset, batch_size=batch_size, shuffle=True,
														   num_workers=19,
														   collate_fn=my_collate_fn,
														   drop_last=(representation_loss == "contrastive"))
							nb_iter_tmp = min(len(mil_train_ndataset), len(mil_train_adataset)) # the number of elements that will be seen during an epoch, for MIL
					else:
						if nb_clips_per_movie == "all":
							# a. standard loader
							my_sampler = FullMovieDifferentialSampler(train_dataset=train_dataset,
																	  nb_movies_per_batch=nb_movies_per_batch)
							# From there we construct the loader
							train_loader = DataLoader(train_dataset, num_workers=19,
													  collate_fn=my_collate_fn,
													  batch_sampler=my_sampler,
													  drop_last=False)
							# b. mil normal loader
							if "MIL" in representation_loss:
								my_mil_nsampler = FullMovieDifferentialSampler(train_dataset=mil_train_ndataset,
																		  nb_movies_per_batch=nb_movies_per_batch)
								mil_train_nloader = DataLoader(mil_train_ndataset, num_workers=19,
														  collate_fn=my_collate_fn,
														  batch_sampler=my_mil_nsampler,
														  drop_last=False)
								# c. mil anormal loader
								my_mil_asampler = FullMovieDifferentialSampler(train_dataset=mil_train_adataset,
																			   nb_movies_per_batch=nb_movies_per_batch)
								mil_train_aloader = DataLoader(mil_train_adataset, num_workers=19,
														  collate_fn=my_collate_fn,
														  batch_sampler=my_mil_asampler,
														  drop_last=False)
								nb_iter_tmp = min(len(my_mil_nsampler), len(my_mil_asampler))  # the number of elements that will be seen during an epoch, for MIL

						else:
							batch_size = nb_movies_per_batch * nb_clips_per_movie
							# a. standard loader
							if "temporal" in representation_architecture:
								my_sampler = TemporalDifferentialSampler(train_dataset=train_dataset,
																		 nb_clips_per_movie=nb_clips_per_movie)
							else:
								my_sampler = DifferentialSampler(train_dataset=train_dataset,
																 nb_clips_per_movie=nb_clips_per_movie)
							# From there we construct the loader
							train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=19,
													  collate_fn=my_collate_fn,
													  sampler=my_sampler,
													  drop_last=False,
													  # processing of this kind is performed in the sampler
													  )
							# b. mil normal loader
							if "MIL" in representation_loss:
								my_mil_nsampler = DifferentialSampler(train_dataset=mil_train_ndataset,
																	  nb_clips_per_movie=nb_clips_per_movie)
								mil_train_nloader = DataLoader(mil_train_ndataset, batch_size=batch_size, num_workers=19,
															   collate_fn=my_collate_fn,
															   sampler=my_mil_nsampler,
															   drop_last=False,
															   # processing of this kind is performed in the sampler
															   )
								# c. mil anormal loader
								my_mil_asampler = DifferentialSampler(train_dataset=mil_train_adataset,
																	  nb_clips_per_movie=nb_clips_per_movie)
								mil_train_aloader = DataLoader(mil_train_adataset, batch_size=batch_size, num_workers=19,
															   collate_fn=my_collate_fn,
															   sampler=my_mil_asampler,
															   drop_last=False,
															   # processing of this kind is performed in the sampler
															   )
								nb_iter_tmp = min(len(my_mil_nsampler), len(my_mil_asampler))
								# the number of elements that will be seen during an epoch, for MIL
								# Here we use the samplers to define it,
								# because differential sampling may drop more than the last batch to ensure nb_clips_per_movie

					# We retrieve the train ids of this fold
					train_imdbs = set(ALL_MOVIES).difference(folds_info[fold_idx]['validation']).difference(
						folds_info[fold_idx]['test'])
					fold_train_ids = sorted([imdb_to_id[imdb_k] for imdb_k in train_imdbs])
					print(f"The train ids of this fold are : {fold_train_ids}")

					# For DANN, we define the mapping from movie id to domain, for the train set
					nb_domains=None # for easy generalization of code to non-DANN models
					if "dann" in representation_architecture:
						if "dann_binary" in representation_architecture:
							nb_domains = 2
							nb_ids = len(fold_train_ids)
							shuffle(fold_train_ids)
							mapping_domain = lambda id: 0 if id in fold_train_ids[:(nb_ids // 2)] else 1
						elif "dann_by_prop" in representation_architecture:
							nb_domains = int(representation_architecture.split("_")[-1])
							nb_ids = len(fold_train_ids)
							assert nb_domains <= nb_ids
							id_to_positive_prop = train_dataset.id_to_positive_prop
							# We only consider the proportion of train_ids
							train_id_to_positive_prop = {k: v for k, v in id_to_positive_prop.items() if k in fold_train_ids}
							# We sort the ids according to their proportion of positive
							sorted_ids = sorted(train_id_to_positive_prop, key=train_id_to_positive_prop.get)
							# We split the ids into buckets (that are as balanced as possible)
							indices = np.linspace(0, nb_ids, num=nb_domains + 1, endpoint=True, dtype=int)
							domain_to_ids = {i: sorted_ids[a:b] for i, (a, b) in enumerate(zip(indices, indices[1:]))}
							mapping_dict = {int(movie_id): domain for domain, movie_ids in domain_to_ids.items() for movie_id in movie_ids}  # keys and values are integers
							mapping_domain = lambda id: mapping_dict.get(id)
						else:
							nb_domains = len(fold_train_ids)
							# N.B. : we need to define this mapping dictionary beforehand, because there is no guarantee that each batch will contain all ids
							mapping_dict = {int(k): i for i, k in enumerate(fold_train_ids)}  # keys and values are integers
							mapping_domain = lambda id: mapping_dict.get(id)

					# For checkpointing
					checkpoint_dir = f"{CHECKPOINTS_DIR}/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{train_annotator}"
					train_annotator_backbone = train_annotator if train_annotator_backbone is None else train_annotator_backbone
					checkpoint_backbone_dir = (f"{CHECKPOINTS_DIR}/{modality}/"
											   f"{representation_backbone['exp_name'] if representation_backbone is not None and 'exp_name' in representation_backbone.keys() else exp_name}/"
											   f"{representation_backbone['pool'] if representation_backbone is not None and 'pool' in representation_backbone.keys() else pool}{representation_backbone['spec_str'] if representation_backbone is not None and 'spec_str' in representation_backbone.keys() else spec_str}/"
											   f"{representation_backbone['dataset_name'] if representation_backbone is not None and 'dataset_name' in representation_backbone.keys() else dataset_name}/"
											   f"{train_annotator_backbone}")
					if not os.path.exists(checkpoint_dir):
						os.makedirs(checkpoint_dir)
					global_model_path_representation = f"{checkpoint_dir}/best_ckpt_seed_{seed}_fold_{fold_idx}_repr_{repr_str}{pseudo_labeling_str}.pt"
					last_global_model_path_representation = f"{checkpoint_dir}/best_ckpt_seed_{seed}_fold_{fold_idx}_repr_{repr_str}.pt" if pseudo_labeling else None
					# Models from successive training cycles are identified by their training_cycle_id ; however,
					# it may be useful to keep track of the last updated model, regardless of the training_cycle id
					global_model_path_classification = f"{checkpoint_dir}/best_ckpt_seed_{seed}_fold_{fold_idx}_classif_{type_classif_str}_on_{repr_str}{pseudo_labeling_str}.pt"
					existing_representation_model = os.path.exists(global_model_path_representation)
					existing_classification_model = os.path.exists(global_model_path_classification)
					if representation_backbone is not None:
						backbone_repr_str, backbone_type_representation = get_repr_strs(**representation_backbone)
					else:
						backbone_repr_str = "dummy" # will not be used
					backbone_path = f"{checkpoint_backbone_dir}/best_ckpt_seed_{seed}_fold_{fold_idx}_repr_{backbone_repr_str}.pt"
					#backbone_path will only be used by "aposteriori_differential" models and "frozen_context_differential_ce"

					device = set_gpu(gpu_index=gpu_index)

					writer = SummaryWriter(f"{PROJECT_DIR}/tensorboard_logs/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{train_annotator}/"
										   f"repr_{repr_str}/seed_{seed}/fold_{fold_idx}{pseudo_labeling_str}")

					# TRAINING HYPERPARAMETERS
					print(f"SETTING HYPERPARAMETERS...")
					# 1. Warm-up steps
					steps_per_epoch = int(len(train_dataset) / batch_size)
					steps_per_epoch_class = int(len(train_dataset) / batch_size_class)
					# depends on the dataset and batch size (total rows / batch size -> around 500 / 16)
					warmup_steps = int(0.1 * total_epochs * steps_per_epoch)
					warmup_steps_class = int(0.1 * total_epochs_class * steps_per_epoch_class)

					# 2. Losses
					if global_contrastive:
						contrastive_loss = GlobalSupConLoss(contrastive_type=contrastive_type, temperature=temperature)
					else:
						contrastive_loss = IntraClusterSupConLoss(contrastive_type=contrastive_type,
																  temperature=temperature, weighted=weighted_contrastive)
					# Standard CE loss for classification
					if classif_info['type_classif'] == 'several_concepts':
						if balancing:
							pos_weight = torch.Tensor(train_dataset.pos_weight).to(device)
							print(f"We perform multi-label balancing using pos_weight = {pos_weight}")
							loss_ce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
						else:
							loss_ce = nn.BCEWithLogitsLoss()
					else:
						if num_output_neurons > 1: # ie. multiclass
							# Whether or not the train dataset is balanced, we do not weight the loss
							loss_ce = nn.CrossEntropyLoss()
							loss_ce_val = nn.CrossEntropyLoss()
						else:
							loss_ce_val = nn.BCEWithLogitsLoss()  # anyway for validation, we want an unweighted loss
							if balancing:
								loss_ce = nn.BCEWithLogitsLoss()
							else:
								loss_ce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([train_dataset.pos_weight]).to(device))

					# A few remarks about the generalization of the code:
					# - both nn.BCEWithLogitsLoss() and nn.CrossEntropyLoss() expect logits, ie. unnormalized logits
					# - nn.BCEWithLogitsLoss() expects input and targets to be of the same shape, while
					#   nn.CrossEntropyLoss() expects (basically) input of shape [N, C] and target of shape [N]
					# - nn.BCEWithLogitsLoss() expects float target while nn.CrossEntropyLoss() expects LongTensor target
					# So : I will handle the type when loading the targets ; and handle the shape when computing the loss

					# Loss for domain classification, for DANN
					domain_loss_ce = nn.CrossEntropyLoss()

					# Concept loss for CEM models
					concept_loss_ce = nn.BCELoss()

					# 3. Learning rate
					learning_rate = 1e-4
					# 4. Epochs and early stopping
					min_epochs = min_epochs
					max_epochs = total_epochs
					patience = 10  # after patience runs without any improvement (regarding val_loss), training is stopped

					# And similarly for the classification head
					learning_rate_class = 1e-4
					min_epochs_class = min_epochs_class
					max_epochs_class = total_epochs_class
					patience_class = 10

					# We also define the "scheduling decay" that will be used for scheduled_double_differential_ce
					#my_scheduling = partial(linear_decay, lambda_f=0., t_decay=13)
					my_scheduling = partial(exponential_decay, lambda_f=0., t_decay=13)
					# because in practice training lasts around 20 epochs, which is way less than max_epochs = typically 100

					# MODEL
					print(f"CREATING MODEL...")
					# For representation
					vit_model = load_representation_model(representation_architecture=representation_architecture,
														  global_model_path_representation=None,
														  device=device,
														  # MODEL HYPERPARAMETERS
														  feature_dim=feature_dim, max_seq_len=max_seq_len,
														  dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
														  pool=pool, dim_head=dim_head,
														  dropout=dropout, emb_dropout=emb_dropout,
								  # To DISTINGUISH BETWEEN BINARY AND MULTICLASS CASE
								  num_output_neurons=num_output_neurons,
														  # SPECIFIC TO SOME MODELS ONLY
														  nb_domains=nb_domains,
														  backbone_path=backbone_path,
														  modality=modality)

					if representation_loss == "target_embedding":
						target_repr_str, target_type_representation = get_repr_strs(**target_model)
						global_target_path_representation = f"{checkpoint_dir}/best_ckpt_seed_{seed}_fold_{fold_idx}_repr_{target_repr_str}.pt"
						# N.B. : as it is implemented for now, the target model only specified the "type of model" ;
						# the one that is loaded is necessarily the one corresponding to the same fold, train annotator, etc.
						target_vit_model = load_representation_model(representation_architecture=target_model['representation_architecture'],
															  global_model_path_representation=global_target_path_representation,
															  device=device,
															  # MODEL HYPERPARAMETERS
															  feature_dim=feature_dim, max_seq_len=max_seq_len,
															  dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
															  pool=pool, dim_head=dim_head,
															  dropout=dropout, emb_dropout=emb_dropout,
								  # To DISTINGUISH BETWEEN BINARY AND MULTICLASS CASE
								  num_output_neurons=num_output_neurons,
															  # SPECIFIC TO SOME MODELS ONLY
															  nb_domains=nb_domains,
															  backbone_path=backbone_path,
																	 modality=modality)
						target_vit_model.eval() # we want to keep it frozen

					if print_nb_parameters:
						print(vit_model)
						#count_parameters(vit_model.named_parameters())
						count_all_parameters(vit_model.named_parameters())
						#init_stats_df = summarize_initial_weights(vit_model, as_dataframe=True, print_topk=30)
						#print(f"weights of the NN :\n{init_stats_df.to_string()}")
						#init_stats_df.to_csv(f"{PROJECT_DIR}/init_stats_vit_seed_{seed}_fold_{fold_idx}.csv", index=False,)

					# We also build the associated optimizer and scheduler
					if "dann" in representation_architecture:
						optimizer = optim.AdamW(vit_model.parameters(), lr=learning_rate, weight_decay=0.00005)
						optimizer = PCGrad(optimizer)
						scheduler = ReduceLROnPlateau(optimizer._optim, 'min', patience=5, factor=0.1)
					elif representation_architecture not in ["xclipemb"]:
						optimizer = optim.AdamW(vit_model.parameters(), lr=learning_rate, weight_decay=0.00005)
						scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

					# REPRESENTATION MODEL TRAINING
					epoch = 0
					global_step = 0
					best_vloss = 1_000_000.  # this variable will be used for checkpointing the best model (based on validation loss)
					corresponding_loss = 1_000_000.  # we keep track of the corresponding training loss
					impatience = 0
					val_losses = []
					early_stopped = False
					target_achieved = False
					if save_training_dynamics or train_on_target:
						# The difference between the actual model we are training here, and the one to obtain ttl,
						#is that by definition the latter was not computed using target_train_loss, or pseudo-labeling, so:
						ttl_repr_str, ttl_type_representation = get_repr_strs(representation_architecture, representation_loss,
																	  global_contrastive, contrastive_type,
																	  weighted_contrastive, temperature,
																	  nb_movies_per_batch, nb_clips_per_movie,
																	  nb_segment, length_segment,
																	  batch_size, train_annotator_backbone,
																	  representation_backbone, target_model,
																	  train_on_target=False, pseudo_labeling=False,
																	  confidence_based_pseudo_labeling=False, budget=None,
																			  fraction_train=fraction_train,
																			  randint_train=randint_train
																	  )

						ttl_dataset_name = "mobygaze_dataframe_soft_visual_corrected_unbalanced" # instead of unbalanced_dataset_name, for now
						ttl_directory = f'{PROJECT_DIR}/target_train_losses/modality/{modality}/{exp_name}/{pool}/{ttl_dataset_name}/{train_annotator}'
						json_name = f"{ttl_repr_str}_target_train_losses"
						if not os.path.exists(f'{ttl_directory}/{json_name}.json'):
							raise Exception(f'{ttl_directory}/{json_name}.json does not exist : '
											f'first compute target_train_losses using save_target_train_losses argument')
						with open(f'{ttl_directory}/{json_name}.json', 'r') as file:
							target_train_loss = min(json.load(file)['target_train_losses'])
							print(f"target_train_loss : {target_train_loss}")

					else:
						target_train_loss = (-np.inf)
					# target_achieved is used for computing training dynamics or train_on_target only, where
					# (because all data is train we need to know when to stop) training is finished once we achieve a specified target
					# Thanks to overfitting, we are almost guaranteed to achieve the loss target in the worst case;
					# just in case we specify max_epochs
					# Anyway the whole point of train target is that we shouldn't even need to overfit to achieve this score

					train_representation = True
					if representation_architecture == "xclipemb":
						train_representation = False
						print(f"\nNO TRAIN OF REPRESENTATION : WE USE X-CLIP EMBEDDINGS")
					elif "MIL" in representation_loss:
						# We do not train MIL in this loop, as its mechanism is fundamentally different
						train_representation = False
					elif untrained:
						train_representation = False
						print(f"UNTRAINED OPTION CHOSEN : THE UNTRAINED MODEL IS SAVED AT "
							  f"{global_model_path_representation}")
						saved_parameters = {k: v for k, v in vit_model.state_dict().items() if not "backbone" in k}
						torch.save(saved_parameters, global_model_path_representation)
					else:
						if use_trained_representation_models and existing_representation_model:
							train_representation = False
							print(f"\nNO TRAIN OF REPRESENTATION : WE USE THE ALREADY TRAINED MODEL SAVED AT "
								  f"{global_model_path_representation}")

					if train_representation:
						print(f"\nREPRESENTATION MODEL TRAINING...")
						# TRAINING OF REPRESENTATION (and possibly classification, if CE/DANN)
						while (not early_stopped and not target_achieved) and epoch < max_epochs:
							print(f"\nEPOCH {epoch} :")
							sampling_rate = my_scheduling(epoch)
							if representation_architecture == "scheduled_double_differential_vit":
								print(f"sampling_rate : {sampling_rate}")

							# Make sure gradient tracking is on
							vit_model.train(True)
							running_loss = 0.
							running_base_loss = 0.
							running_concept_loss = 0.
							running_loss_domain = 0.

							# If we save training dynamics, we will save train logits and guids
							full_binary_targets_np = []
							full_train_logits_np = []
							full_guids_np = []

							for i_train, batch in enumerate(train_loader):
								# print(f"batch {i_train} ; global step {global_step}")
								# print(f"global_step : {global_step} ; current lr : {optimizer.param_groups[0]['lr']}")
								# Proper training
								visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = batch

								# N.B. : for multi-modal case, visual_features and visual_mask are actually tuples
								# (of standard features and mask)
								# one_hot_concepts : [bs, num_concepts], where num_concepts depends on the considered modality

								domain_label = torch.clone(ids)

								if "dann" in representation_architecture:
									domain_label.apply_(mapping_domain)  # we do apply_ before it gets to cpu

								clip_label, one_hot_concepts, ids = (
									clip_label.to(device),
									one_hot_concepts.to(device),
									ids.to(device))
								if multimodal:
									for i_mod in range(len(visual_features)):
										visual_features[i_mod] = visual_features[i_mod].to(device)
										visual_mask[i_mod] = visual_mask[i_mod].to(device)
								else:
									visual_features = visual_features.to(device)
									visual_mask = visual_mask.to(device)

								domain_label = domain_label.to(device)
								binary_targets = clip_label.float().to(device) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else clip_label.type(torch.LongTensor).to(device)
								# N.B. : Somehow changing type with type() may change the device, so we need to specify .to(device) again
								binary_targets_for_loss = binary_targets.unsqueeze(1) if num_output_neurons == 1 else binary_targets

								#if i_train == 0:
									#print(f"visual_features.shape (for one train batch) : {visual_features.shape}")

								if "representation" in representation_architecture: # representation_vit or representation_mlp
									if representation_architecture == "representation_vit":
										representations = vit_model(visual_features, visual_mask)
									elif representation_architecture == "representation_mlp":
										representations = vit_model(visual_features)
									else:
										raise Exception(f"representation_architecture should be representation_vit or "
														f"representation_mlp : here it is '{representation_architecture}'")

									if representation_loss =="contrastive":
										all_losses = contrastive_loss(
											{'feats': representations, 'labels': binary_targets, 'ids': ids})
										loss = all_losses['returned_loss']
										optimizer.zero_grad()  # Zero your gradients for every batch!
										loss.backward(retain_graph=True)  # retain_graph=True is necessary for simultaneous training

									elif "target_embedding" in representation_loss:
										with ((torch.no_grad())):
											if "vit" in target_model['representation_architecture'] or "mlp" in target_model['representation_architecture']:
												target_representations, target_logits, c_sem = standard_inference(
													representation_architecture=target_model['representation_architecture'],
													vit_model=target_vit_model, visual_features=visual_features,
													visual_mask=visual_mask, ids=ids, clip_label=clip_label,
													one_hot_concepts= one_hot_concepts, train=True and randint_train
													)
											elif "dann" in target_model['representation_architecture']:
												target_representations, target_classif_logits, target_domain_logits = dann_inference(
													representation_architecture=target_model['representation_architecture'],
													vit_model=target_vit_model, visual_features=visual_features,
													visual_mask=visual_mask, ids=ids, clip_label=clip_label)
											else:
												raise Exception(f"Unkown target architecture : '{target_model['representation_architecture']}'")
										# From there we can compute the cosine distance element-wise : the loss is their sum
										normalized_representations = F.normalize(representations, dim=-1, p=2)
										normalized_target_representations = F.normalize(target_representations, dim=-1, p=2)
										similarities_to_target = (normalized_representations @ normalized_target_representations.T).diag()
										loss = 1 - torch.mean(similarities_to_target)
										optimizer.zero_grad()  # Zero your gradients for every batch!
										loss.backward()

								elif "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
									representations, logits, c_sem = standard_inference(representation_architecture=representation_architecture,
																		   vit_model=vit_model, visual_features=visual_features,
																		   visual_mask=visual_mask, ids=ids, clip_label=clip_label,
																		   sampling_rate=sampling_rate,
																						one_hot_concepts= one_hot_concepts, train=True and randint_train)

									if pseudo_labeling: # the loss if the weighted sum of "labeled" loss and "unlabeled" loss
										pseudo_label_mask = np.isin(guids, pseudo_labels_guids)
										labeled_mask = np.bitwise_not(pseudo_label_mask)
										labeled_logits, labeled_targets = logits[labeled_mask], binary_targets_for_loss[labeled_mask]
										labeled_loss = loss_ce(labeled_logits, labeled_targets)
										pseudo_logits, pseudo_targets = logits[pseudo_label_mask], binary_targets_for_loss[pseudo_label_mask]
										pseudo_loss = loss_ce(pseudo_logits, pseudo_targets)
										# If one of the set (labeled, pseudo) is empty, the corresponding loss is nan
										if len(labeled_logits) > 0:
											if len(pseudo_logits) > 0:
												loss = labeled_loss + pseudo_labeling_weight * pseudo_loss
											else:
												loss = labeled_loss
										else:
											loss = pseudo_labeling_weight * pseudo_loss

									else:
										loss = loss_ce(logits, binary_targets_for_loss)
										if representation_architecture[:4] == "cem_" or representation_architecture[:13] == "backbone_cem_":
											base_loss = loss.clone()
											concept_loss = concept_loss_ce(c_sem, one_hot_concepts)
											loss += concept_loss_weight * concept_loss

									optimizer.zero_grad()  # Zero your gradients for every batch!
									loss.backward()

									# We compute related variables that can be used for computing metrics
									# See calculate_tb_metrics method of the Lightning module if I want to compute them
									#probabilities = torch.sigmoid(logits) if num_output_neurons==1 or classif_info['type_classif'] == 'several_concepts' else torch.softmax(logits, dim=-1)
									#pred_label = (probabilities >= 0.5).int() if num_output_neurons==1 else torch.argmax(probabilities, dim=-1).int()
									#binary_targets_np = binary_targets.detach().cpu().numpy()
									#pred_label_np = pred_label.detach().cpu().numpy()
									#probabilities_np = probabilities.detach().cpu().numpy()
									# Tensorboard logging : for training, it is every step/batch
									# N.B. : We only log the loss : indeed, when our loader has keep_last=True,
									# it may happen that our batch contains only 1 or 2 examples, so
									# all the batch contains the same class and some classification metrics are ill-defined

									if save_training_dynamics:
										binary_targets_np = binary_targets.detach().cpu().numpy()
										logits_np = logits.detach().cpu().numpy()
										# To match the structure of the results that are saved with dataset cartography, where logits for all classes are saved,
										# we artificially compute the logits corresponding to the negative class ;
										# probabilities were obtained by applying sigmoid to (positive) logits, which is equivalent to
										# softmax where the second logit is zero
										if num_output_neurons == 1:
											negative_logits_np = np.zeros(logits_np.shape)
											train_logits_np = np.hstack((negative_logits_np, logits_np))
										else:
											train_logits_np = logits_np
										# Now we add those metrics to the list over batches
										full_binary_targets_np.append(binary_targets_np)
										full_train_logits_np.append(train_logits_np)
										full_guids_np.append(guids)

								elif "dann" in representation_architecture:
									representations, classif_logits, domain_logits = dann_inference(representation_architecture=representation_architecture,
																		   vit_model=vit_model, visual_features=visual_features,
																		   visual_mask=visual_mask, ids=ids, clip_label=clip_label)
									loss = loss_ce(classif_logits, binary_targets_for_loss)
									loss_domain_dann = domain_loss_ce(domain_logits, domain_label)
									losses = [loss, loss_domain_dann]
									optimizer.pc_backward(losses)

								else:
									raise Exception(f"Unsupported combo of representation_architecture / representation_loss : "
													f"{representation_architecture} / {representation_loss}")

								# I moved the model .step() call here so that all the .step() happen after all the .backward() :
								# else it might throw an error
								optimizer.step()

								warm_up_lr_adaptation(global_step, warmup_steps, learning_rate, optimizer,
													  optimizer_is_attribute="dann" in representation_architecture)

								# Gather data and report
								running_loss += loss.item()
								if representation_architecture[:4] == "cem_" or representation_architecture[:13] == "backbone_cem_":
									running_base_loss += base_loss.item()
									running_concept_loss += concept_loss.item()
								if "dann" in representation_architecture:
									running_loss_domain += loss_domain_dann.item()

								global_step += 1

							if save_training_dynamics:
								# From all individual batch results, we can construct the full dataset results :
								full_binary_targets_np = np.concatenate(full_binary_targets_np)
								full_train_logits_np = np.concatenate(full_train_logits_np)
								full_guids_np = np.concatenate(full_guids_np)
								# We save the full training dynamics, similarly to dataset cartography
								training_dynamics_dir = f"{PROJECT_DIR}/training_dynamics/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{train_annotator}/repr_{repr_str}_seed_{seed}"
								if not os.path.exists(training_dynamics_dir):
									os.makedirs(training_dynamics_dir)
								log_training_dynamics(logging_dir=training_dynamics_dir,
													  epoch=epoch,
													  train_ids=full_guids_np.tolist(),
													  train_logits=full_train_logits_np.tolist(),
													  train_golds=full_binary_targets_np.astype(int).tolist())

							avg_loss = running_loss / (i_train + 1)
							writer.add_scalar(f"loss/train", avg_loss, global_step=epoch)
							print(f"Model train loss : {avg_loss}")
							if "dann" in representation_architecture:
								avg_loss_domain = running_loss_domain / (i_train + 1)
								writer.add_scalar(f"domain_loss/train", avg_loss_domain, global_step=epoch)
								print(f"Model train domain loss : {avg_loss_domain}")
							if representation_architecture[:4] == "cem_" or representation_architecture[:13] == "backbone_cem_":
								avg_base_loss = running_base_loss / (i_train + 1)
								avg_concept_loss = running_concept_loss / (i_train + 1)
								writer.add_scalar(f"base_loss/train", avg_base_loss, global_step=epoch)
								writer.add_scalar(f"concept_loss/train", avg_concept_loss, global_step=epoch)
								print(f"Train : Base (CE) loss : {avg_base_loss} / Concept loss : {avg_concept_loss}")

							if avg_loss < target_train_loss:
								target_achieved = True

							# VALIDATION
							# It is mainly required for early stopping, so it is not necessary for train_on_target
							if len(val_loader) > 0 and not train_on_target:
								running_vloss = 0.0
								running_base_vloss = 0.0
								running_concept_vloss = 0.0
								vit_model.eval()

								# Disable gradient computation and reduce memory consumption.
								with ((torch.no_grad())):
									list_representations = []
									list_ids = []
									list_probabilities = []
									list_binary_targets = []
									list_pred_label = []
									for i_val, vbatch in enumerate(val_loader):
										# N.B. : Note that, as val_loader is implemented for now,
										# the validation dataset actually corresponds to only one batch
										visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = vbatch

										clip_label, one_hot_concepts, ids = (
											clip_label.to(device),
											one_hot_concepts.to(device),
											ids.to(device))
										if multimodal:
											for i_mod in range(len(visual_features)):
												visual_features[i_mod] = visual_features[i_mod].to(device)
												visual_mask[i_mod] = visual_mask[i_mod].to(device)
										else:
											visual_features = visual_features.to(device)
											visual_mask = visual_mask.to(device)

										binary_targets = clip_label.float().to(device) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else clip_label.type(torch.LongTensor).to(device)
										# N.B. : Somehow changing type with type() may change the device, so we need to specify .to(device) again
										binary_targets_for_loss = binary_targets.unsqueeze(1) if num_output_neurons == 1 else binary_targets

										if "representation" in representation_architecture:  # representation_vit or representation_mlp
											if representation_architecture == "representation_vit":
												representations = vit_model(visual_features, visual_mask)
											elif representation_architecture == "representation_mlp":
												representations = vit_model(visual_features)
											else:
												raise Exception(f"representation_architecture should be representation_vit or "
																f"representation_mlp : here it is '{representation_architecture}'")

											if representation_loss == "contrastive":
												all_vlosses = contrastive_loss(
													{'feats': representations, 'labels': binary_targets, 'ids': ids})
												vloss = all_vlosses['returned_loss']
												# Print logging
												print(f"Validation contrastive losses : {all_vlosses}")

											elif "target_embedding" in representation_loss:
												if "vit" in target_model['representation_architecture'] or "mlp" in target_model['representation_architecture']:
													target_representations, target_logits, c_sem = standard_inference(
														representation_architecture=target_model[
															'representation_architecture'],
														vit_model=target_vit_model, visual_features=visual_features,
														visual_mask=visual_mask, ids=ids, clip_label=clip_label,
													)
												elif "dann" in target_model['representation_architecture']:
													target_representations, target_classif_logits, target_domain_logits = dann_inference(
														representation_architecture=target_model[
															'representation_architecture'],
														vit_model=target_vit_model, visual_features=visual_features,
														visual_mask=visual_mask, ids=ids, clip_label=clip_label)
												else:
													raise Exception(
														f"Unkown target architecture : '{target_model['representation_architecture']}'")
												# From there we can compute the cosine distance element-wise : the loss is their sum
												normalized_representations = F.normalize(representations, dim=-1, p=2)
												normalized_target_representations = F.normalize(target_representations, dim=-1,p=2)
												similarities_to_target = (normalized_representations @ normalized_target_representations.T).diag()
												vloss = 1 - torch.mean(similarities_to_target)


										elif "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
											representations, logits, c_sem = standard_inference(representation_architecture=representation_architecture,
																				   vit_model=vit_model,
																				   visual_features=visual_features,
																				   visual_mask=visual_mask, ids=ids,
																				   clip_label=None if no_cheating_val else clip_label,
																				   sampling_rate=0.)
											vloss = loss_ce(logits, binary_targets_for_loss)
											if representation_architecture[:4] == "cem_" or representation_architecture[:13] == "backbone_cem_":
												base_vloss = vloss.clone()
												concept_vloss = concept_loss_ce(c_sem, one_hot_concepts)
												vloss += concept_loss_weight * concept_vloss
											# We compute related variables that can be used for computing metrics
											# See calculate_tb_metrics method of the Lightning module if I want to compute them
											probabilities = torch.sigmoid(logits) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.softmax(logits, dim=-1)
											pred_label = (probabilities >= 0.5).int() if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.argmax(probabilities, dim=-1).int()
											list_probabilities.append(probabilities)
											list_pred_label.append(pred_label)

										elif "dann" in representation_architecture:
											representations, classif_logits, domain_logits = dann_inference(
												representation_architecture=representation_architecture,
												vit_model=vit_model, visual_features=visual_features,
												visual_mask=visual_mask, ids=ids, clip_label=None if no_cheating_val else clip_label)
											loss_classif_dann = loss_ce(classif_logits, binary_targets_for_loss)
											# Note that the domain classifier of the DANN model was constructed for as many "domain classes" as there are
											# training ids. In particular, those training ids are different and in different number from
											# the validation ids ; so it would make little sense to consider the validation domain loss.
											vloss = loss_classif_dann
											# We compute related variables that can be used for computing metrics
											# See calculate_tb_metrics method of the Lightning module if I want to compute them
											probabilities = torch.sigmoid(classif_logits) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.softmax(logits, dim=-1)
											pred_label = (probabilities >= 0.5).int() if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.argmax(probabilities, dim=-1).int()
											list_probabilities.append(probabilities)
											list_pred_label.append(pred_label)

										else:
											raise Exception(
												f"Unsupported combo of representation_architecture / representation_loss : "
												f"{representation_architecture} / {representation_loss}")

										running_vloss += vloss.item()
										if representation_architecture[:4] == "cem_" or representation_architecture[:13] == "backbone_cem_":
											running_base_vloss += base_vloss.item()
											running_concept_vloss += concept_vloss.item()

										list_representations.append(representations)
										list_ids.append(ids)
										list_binary_targets.append(clip_label)

								# Average validation loss and validation metrics

								avg_vloss = running_vloss / (i_val + 1)
								print(f"Model validation loss : {avg_vloss}")
								if representation_architecture[:4] == "cem_" or representation_architecture[:13] == "backbone_cem_":
									avg_base_vloss = running_base_vloss / (i_val + 1)
									avg_concept_vloss = running_concept_vloss / (i_val + 1)
									writer.add_scalar(f"concept_loss/val", avg_concept_vloss, global_step=epoch)
									writer.add_scalar(f"base_loss/val", avg_base_vloss, global_step=epoch)
									print(f"Validation : Base (CE) loss : {avg_base_vloss} / Concept loss : {avg_concept_vloss}")

								# We gather all validation batches
								binary_targets = torch.cat(list_binary_targets)
								binary_targets_np = binary_targets.detach().cpu().numpy()
								ids = torch.cat(list_ids)
								representations = torch.cat(list_representations)

								if ("vit" in representation_architecture or "dann" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture) and representation_architecture not in ["representation_vit", "representation_mlp"]:
									probabilities = torch.cat(list_probabilities)
									pred_label = torch.cat(list_pred_label)
									pred_label_np = pred_label.detach().cpu().numpy()
									probabilities_np = probabilities.detach().cpu().numpy()
									# Tensorboard logging : for validation, it is every epoch
									if num_output_neurons == 1:
										val_metrics = calculate_tb_metrics(avg_vloss, binary_targets_np, pred_label_np,
																		   probabilities_np,
																				 num_output_neurons=num_output_neurons)
										val_metrics = {k: round(float(v), 3) for k, v in val_metrics.items()}
										for k, v in val_metrics.items():
											writer.add_scalar(f"{k}/val", v, global_step=epoch)
										# Print logging
										print(f"Validation CE metrics :\n{val_metrics}")
									else:
										writer.add_scalar(f"loss/val", avg_vloss, global_step=epoch)
								# Even if the ViT/DANN-model is only used for representation here, it is still useful
								# for comparison to print its classification metrics

								val_losses.append(avg_vloss)

								# In all cases we have "representations" that we can evaluate
								if evaluate_representation:
									all_representation_val_metrics = test_representations_various_probers(
										{'feats': representations, 'labels': binary_targets, 'ids': ids},
										type_probers=type_probers)
									print(f"Validation representation metrics :")
									for type_prober, representation_val_metrics in zip(type_probers,
																					   all_representation_val_metrics):
										for k, v in representation_val_metrics['average'].items():
											writer.add_scalar(f"average_{k}/val", v, global_step=epoch)
										# Print logging
										print(f"Prober {type_prober} :"
											  f"\nAverage (across movie clusters) : {representation_val_metrics['average']}"
											  f"\nGlobal : {representation_val_metrics['global']}")

								# Tracks best performance, and save the model's state
								if avg_vloss < best_vloss:
									print(f"New best model (epoch {epoch})")
									best_vloss = avg_vloss
									corresponding_loss = avg_loss
									impatience = 0
									if save_model:
										saved_parameters = {k:v for k, v in vit_model.state_dict().items() if not "backbone" in k}
										torch.save(saved_parameters, global_model_path_representation)
										print(f"Saved model at {global_model_path_representation}")
										if last_global_model_path_representation is not None:
											torch.save(saved_parameters, last_global_model_path_representation)
											print(f"Saved 'last-updated' model at {last_global_model_path_representation}")
								else:
									impatience += 1

								early_stopped = (epoch + 1 >= min_epochs) and (impatience >= patience)
								if early_stopped:
									print(f"Early stopping\nSuccessive val losses : {[round(vl, 2) for vl in val_losses]}")

								scheduler.step(avg_vloss)

							# In case there is no validation set (typically save_training_dynamics or target_train_loss), we still need to save the model
							else:
								if save_model:
									saved_parameters = {k: v for k, v in vit_model.state_dict().items() if not "backbone" in k}
									torch.save(saved_parameters, global_model_path_representation)
									print(f"Saved model at {global_model_path_representation}")
									if last_global_model_path_representation is not None:
										torch.save(saved_parameters, last_global_model_path_representation)
										print(f"Saved 'last-updated' model at {last_global_model_path_representation}")

							epoch += 1

						annotator_target_train_losses.append(corresponding_loss)

					# Here the specific training loop for MIL
					if "MIL" in representation_loss:
						assert num_output_neurons == 1 # MIL is designed for binary classification only
						if use_trained_representation_models and existing_representation_model:
							print(f"\nNO TRAIN OF REPRESENTATION : WE USE THE ALREADY TRAINED MODEL SAVED AT "
								  f"{global_model_path_representation}")
						else:
							print(f"\nMIL ({representation_architecture}) MODEL TRAINING...")
							# TRAINING OF MIL
							# N.B. : even if they are a bit far away in the code, global training variables like epoch, impatience, etc. are working properly

							# Because we will iterate simultaneously over two loaders, we need to compute in advance the number of batches
							nb_batch = nb_iter_tmp // batch_size

							while not early_stopped and epoch < max_epochs:
								print(f"\nEPOCH {epoch} :")
								sampling_rate = my_scheduling(epoch)
								if representation_architecture == "scheduled_double_differential_vit":
									print(f"sampling_rate : {sampling_rate}")

								# Make sure gradient tracking is on
								vit_model.train(True)
								running_loss = 0.

								n_iter = iter(mil_train_nloader)
								a_iter = iter(mil_train_aloader)
								for i in range(nb_batch):
									ninput, nvisual_mask, nclip_label, none_hot_concepts, nids = next(n_iter)
									ninput, nvisual_mask, nclip_label, none_hot_concepts, nids = ninput.to(device), nvisual_mask.to(device), nclip_label.to(device), none_hot_concepts.to(device), nids.to(device)
									ainput, avisual_mask, aclip_label, aone_hot_concepts, aids = next(a_iter)
									ainput, avisual_mask, aclip_label, aone_hot_concepts, aids = ainput.to(device), avisual_mask.to(device), aclip_label.to(device), aone_hot_concepts.to(device), aids.to(device)

									#if i == 0:
										#print(f"Shapes in the batch before reshaping:")
										#print(f"ninput.shape : {ninput.shape}")
										#print(f"ainput.shape : {ainput.shape}")
										#print(f"nvisual_mask.shape : {nvisual_mask.shape}")
										#print(f"avisual_mask.shape : {avisual_mask.shape}")
										#print(f"nids.shape : {nids.shape}")

									if "mlp" in representation_architecture: # in other words not sequential
										ninput = ninput.view(-1, feature_dim) # shape (nb_segments_in_the_batch, feature_dim)
										ainput = ainput.view(-1, feature_dim) # shape (nb_segments_in_the_batch, feature_dim)
										nvisual_mask = nvisual_mask.view(-1)
										avisual_mask = avisual_mask.view(-1)
										# We do as if each segment (which in this implementation is 1-clip long) was a full clip
										# In particular, because these new annotation clips are 1-clip-long, we cannot use Transformers architecture
									elif "vit" in representation_architecture:
										ninput = ninput.view(-1, max_seq_len, feature_dim)  # shape (nb_segment of the batch, max_seq_len, feature_dim)
										ainput = ainput.view(-1, max_seq_len, feature_dim)  # shape (nb_segment of the batch, max_seq_len, feature_dim)
										nvisual_mask = nvisual_mask.view(-1, max_seq_len)
										avisual_mask = avisual_mask.view(-1, max_seq_len)
										# Designed for using a Transformers architecture
									else:
										raise Exception(f"representation_architecture for MIL must be ViT or MLP : "
														f"you have '{representation_architecture}'")

									#if i == 0:
										#print(f"Shapes in the batch after reshaping :")
										#print(f"ninput.shape : {ninput.shape}")
										#print(f"ainput.shape : {ainput.shape}")
										#print(f"nvisual_mask.shape : {nvisual_mask.shape}")
										#print(f"avisual_mask.shape : {avisual_mask.shape}")

									input_ = torch.cat((ninput, ainput), 0).to(device) # shape (nb_normal_and_anormal_segments_in_the_batch, feature_dim)
									mask_ = torch.cat((nvisual_mask, avisual_mask), 0).to(device)
									ids_ = torch.cat((nids, aids), 0).to(device)
									clip_label_ = torch.cat((nclip_label, aclip_label), 0).to(device)

									# We also construct the mast at batch level :
									# with fixed length MIL, some elements of the batch may actually be empty
									segments_mask_ = mask_ if len(mask_.shape) == 1 else torch.any(mask_, dim=1)


									if "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
										representations, scores, c_sem = standard_inference(
											representation_architecture=representation_architecture,
											vit_model=vit_model,
											visual_features=input_,
											visual_mask=mask_, ids=ids_,
											clip_label=clip_label_,
											sampling_rate=sampling_rate,
										one_hot_concepts= one_hot_concepts, train=True and randint_train)
									else:
										raise Exception(
											f"Unsupported combo of representation_architecture / representation_loss : "
											f"{representation_architecture} / {representation_loss}")
									# For now MIL dann is not supported

									# scores has shape (2 * nb_segment * batch_size, 1)
									# N.B. : for dummy segments (for which we still perform inference, for coding convenience),
									# scores are not null (or constant across dummy segments), so we really need to use the mask
									loss = ranking(scores, segments_mask_, batch_size, nb_segment, device)
									# N.B. : the loss does not need the labels thanks to our construction of inputs :
									# the first half corresponds to normal = 0 ; the second half to anomaly = 1

									optimizer.zero_grad()
									loss.backward()
									optimizer.step()

									warm_up_lr_adaptation(global_step, warmup_steps, learning_rate, optimizer,
														  optimizer_is_attribute="dann" in representation_architecture)

									running_loss += loss.item()

									global_step += 1

								avg_loss = running_loss / nb_batch
								writer.add_scalar(f"MIL_loss/train", avg_loss, global_step=epoch)
								print(f"Model train loss : {avg_loss}")

								# VALIDATION
								running_vloss = 0.0
								vit_model.eval()
								with ((torch.no_grad())):
									list_probabilities = []
									list_binary_targets = []
									for i_val, vbatch in enumerate(mil_val_loader):
										input_, vvisual_mask, label, vone_hot_concepts, guids, vids = vbatch
										input_, vvisual_mask, label, vone_hot_concepts, vids = input_.to(device), vvisual_mask.to(device), label.to(device), vone_hot_concepts.to(device), vids.to(device)
										label = label.float() # to avoid CE errors
										# input_ has shape [batch_size, nb_segment, feature_dim] if "mlp_vit"
										# else [batch_size, nb_segment, max_seq_len, feature_dim]
										val_batch_size = len(input_)

										#if i_val == 0:
											#print(f"Shapes in the validation batch before reshaping:")
											#print(f"input_.shape : {input_.shape}")
											#print(f"vvisual_mask.shape : {vvisual_mask.shape}")

										# flattening is necessary for the model to work well
										# For code generalization it is better to do flattening outside of the model
										if "mlp" in representation_architecture: # in other words not sequential
											input_ = input_.view(val_batch_size * nb_segment, feature_dim)
											vvisual_mask = vvisual_mask.view(-1)
										elif "vit" in representation_architecture:
											input_ = input_.view(val_batch_size * nb_segment, max_seq_len, feature_dim)
											vvisual_mask = vvisual_mask.view(val_batch_size * nb_segment, max_seq_len)
										else:
											raise Exception(f"representation_architecture for MIL must be ViT or MLP : "
															f"you have '{representation_architecture}'")

										if "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
											representations, logits, c_sem = standard_inference(
												representation_architecture=representation_architecture,
												vit_model=vit_model,
												visual_features=input_,
												visual_mask=vvisual_mask, ids=vids,
												clip_label=None if no_cheating_val else label,
												sampling_rate=0.)
										else:
											raise Exception(
												f"Unsupported combo of representation_architecture / representation_loss : "
												f"{representation_architecture} / {representation_loss}")
										# For now MIL dann is not supported

										# logits have shape [batch_size * nb_segment, 1]

										#if i_val == 0:
											#print(f"Shapes in the validation batch after reshaping:")
											#print(f"input_.shape : {input_.shape}")
											#print(f"vvisual_mask.shape : {vvisual_mask.shape}")
											#print(f"logits.shape (before reshaping) : {logits.shape}")

										# We return to the original shape
										logits = logits.view(val_batch_size, nb_segment, 1)
										# N.B. : because PyTorch stores tensors in row-major order, we are guaranteed the original order is preserved
										logits = torch.squeeze(logits, 2) # shape [batch_size, nb_segment]
										logit_max, _ = torch.max(logits, dim=1) # shape [batch_size]
										vloss = loss_ce(logit_max, label) # note this loss is CE different from MIL loss
										running_vloss += vloss.item()

										list_probabilities.append(torch.sigmoid(logit_max))
										list_binary_targets.append(label)

								probabilities = torch.cat(list_probabilities)
								binary_targets = torch.cat(list_binary_targets)
								pred_label = torch.argmax(probabilities).int()
								binary_targets_np = binary_targets.detach().cpu().numpy()
								pred_label_np = pred_label.detach().cpu().numpy()
								probabilities_np = probabilities.detach().cpu().numpy()

								# Printing validation metrics :
								avg_vloss = running_vloss / (i_val + 1)
								print(f"Model validation CE loss : {avg_vloss}")
								val_metrics = calculate_tb_metrics(avg_vloss, binary_targets_np, pred_label_np,
																   probabilities_np,
																		 num_output_neurons=num_output_neurons)
								val_metrics = {k: round(float(v), 3) for k, v in val_metrics.items()}
								for k, v in val_metrics.items():
									writer.add_scalar(f"{k}/val", v, global_step=epoch)
								# Print logging
								print(f"Validation metrics :\n{val_metrics}")

								val_losses.append(avg_vloss)

								# Tracks best performance, and save the model's state
								if avg_vloss < best_vloss:
									print(f"New best model (epoch {epoch})")
									best_vloss = avg_vloss
									impatience = 0
									if save_model:
										saved_parameters = {k: v for k, v in vit_model.state_dict().items() if
															not "backbone" in k}
										torch.save(saved_parameters, global_model_path_representation)
										print(f"Saved model at {global_model_path_representation}")
								else:
									impatience += 1

								early_stopped = (epoch + 1 >= min_epochs) and (impatience >= patience)
								if early_stopped:
									print(f"Early stopping\nSuccessive val losses : {[round(vl, 2) for vl in val_losses]}")

								scheduler.step(avg_vloss)
								epoch += 1


					if representation_architecture != "xclipemb":
						del vit_model
					torch.cuda.empty_cache()

					# THE TRAINING OF REPRESENTATION (and possibly classification, if not "representation_vit") IS FINISHED
					print(f"\nSUCCESSFUL TRAINING OF REPRESENTATION")

					# Now, either we are finished, or we can load, once and for all, our best vit model (for this fold)
					vit_model = load_representation_model(representation_architecture=representation_architecture,
														  global_model_path_representation=global_model_path_representation,
														  device=device,
														  # MODEL HYPERPARAMETERS
														  feature_dim=feature_dim, max_seq_len=max_seq_len,
														  dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
														  pool=pool, dim_head=dim_head,
														  dropout=dropout, emb_dropout=emb_dropout,
								  # TO DISTINGUISH BETWEEN BINARY AND MULTICLASS CASE
								  num_output_neurons=num_output_neurons,
														  # SPECIFIC TO SOME MODELS ONLY
														  nb_domains=nb_domains,
														  backbone_path=backbone_path,
														  modality=modality)
					print(f"\nWe load once and for all (in eval mode) the representation model from \n{global_model_path_representation}")
					if vit_model is not None: # for xclipemb
						vit_model.eval()
					if print_nb_parameters:
						print(vit_model)
						#count_parameters(vit_model.named_parameters())
						count_all_parameters(vit_model.named_parameters())
						#init_stats_df = summarize_initial_weights(vit_model, as_dataframe=True, print_topk=30)
						#print(f"weights of the NN :\n{init_stats_df.to_string()}")
						#init_stats_df.to_csv(f"{PROJECT_DIR}/init_stats_vit_seed_{seed}_fold_{fold_idx}.csv", index=False,)

					# EVALUATION OF THE REPRESENTATION ON THE TEST SET (on the best model, not the last !) FOR THIS FOLD
					evaluate_representation_classifier = evaluate_representation_classifier and representation_architecture not in ["xclipemb", "representation_vit"]
					if (evaluate_representation or evaluate_representation_classifier) and len(folds_info[fold_idx]['test']) > 0:
						for test_annotator in test_annotators:
							print(f"\n\n## test_annotator {test_annotator} :\n")
							if "MIL" in representation_loss:
								test_dataset = MILDataset(modality=modality, gt_csv=gt_csv, split="test", fold_idx=fold_idx,
															  folds_info=folds_info, classif_info=classif_info,
															  max_seq_len=max_seq_len,
															  seed=seed, annotator=test_annotator, with_guid=True, with_id=True,
															  is_normal=True, feature_dim=feature_dim, nb_segment=nb_segment,
														  length_segment=length_segment, type_representation=type_representation)
								test_loader = DataLoader(test_dataset, batch_size=len(test_dataset) if "differential" in representation_architecture else batch_size,
														 shuffle=False, num_workers=19,
														 pin_memory=True,
														 collate_fn=my_collate_fn)
							else:
								test_dataset = FeatureDF(modality=modality, gt_csv=gt_csv, split="test", fold_idx=fold_idx,
														 folds_info=folds_info, classif_info=classif_info, max_seq_len=max_seq_len,
														 seed=seed, annotator=test_annotator, with_guid=True, with_id=True,
								  representation_architecture=representation_architecture,
											  specified_visual_feature_dir=specified_visual_feature_dir)
								test_loader = DataLoader(test_dataset, batch_size=len(test_dataset) if "differential" in representation_architecture else batch_size,
														 shuffle=False, num_workers=19,
														 collate_fn=my_collate_fn,)
							# Thanks to shuffle=False, we will be able to store test probabilities directly
							# (without having to recompute the inference order)
							list_logits = []
							list_representations = []
							list_binary_targets = []
							list_ids = []
							with ((torch.no_grad())):
								for i_val, tbatch in enumerate(test_loader):
									visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = tbatch

									clip_label, one_hot_concepts, ids = (
										clip_label.to(device),
										one_hot_concepts.to(device),
										ids.to(device))
									if multimodal:
										for i_mod in range(len(visual_features)):
											visual_features[i_mod] = visual_features[i_mod].to(device)
											visual_mask[i_mod] = visual_mask[i_mod].to(device)
									else:
										visual_features = visual_features.to(device)
										visual_mask = visual_mask.to(device)

									binary_targets = clip_label.float().to(device) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else clip_label.type(torch.LongTensor).to(device)
									# N.B. : Somehow changing type with type() may change the device, so we need to specify .to(device) again
									binary_targets_for_loss = binary_targets.unsqueeze(1) if num_output_neurons == 1 else binary_targets

									if "MIL" in representation_loss:
										test_batch_size = len(visual_features)
										# flattening is necessary for the model to work well
										# For code generalization it is better to do flattening outside of the model
										if "mlp" in representation_architecture:
											visual_features = visual_features.view(test_batch_size * nb_segment, feature_dim)
											visual_mask = visual_mask.view(-1)
										elif "vit" in representation_architecture:
											visual_features = visual_features.view(test_batch_size * nb_segment, max_seq_len, feature_dim)
											visual_mask = visual_mask.view(test_batch_size * nb_segment, max_seq_len)
										else:
											raise Exception(f"Unknown representation_architecture : '{representation_architecture}'")

									# Now we do the inference
									if representation_architecture == "representation_vit":
										logits = torch.zeros(1)  # representation_vit returns no logits
										representations = vit_model(visual_features, visual_mask)
									elif "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
										representations, logits, c_sem = standard_inference(
											representation_architecture=representation_architecture,
											vit_model=vit_model,
											visual_features=visual_features,
											visual_mask=visual_mask, ids=ids,
											clip_label=None if no_cheating_test else clip_label,
											sampling_rate=0.)
									elif "dann" in representation_architecture:
										representations, logits, domain_logits = dann_inference(
											representation_architecture=representation_architecture,
											vit_model=vit_model, visual_features=visual_features,
											visual_mask=visual_mask, ids=ids,
											clip_label=None if no_cheating_test else clip_label)
									elif representation_architecture == "xclipemb":
										logits = torch.zeros(1)  # xclipemb returns no logits
										representations = torch.mean(visual_features, dim=1)
									else:
										raise Exception(f"Unsupported representation_architecture : '{representation_architecture}'")

									# If MIL model, we reshape the results
									if "MIL" in representation_loss:
										logits = logits.view(test_batch_size, nb_segment, 1)
										logits, _ = torch.max(logits, dim=1) # shape [batch_size, 1]
										# Similarly for representations
										representations = representations.view(test_batch_size, nb_segment, -1)
										representations = torch.mean(representations, dim=1)  # shape [batch_size, repr_dim]

									list_logits.append(logits)
									list_representations.append(representations)
									list_binary_targets.append(binary_targets)
									list_ids.append(ids)

							logits = torch.cat(list_logits)
							representations = torch.cat(list_representations)
							binary_targets = torch.cat(list_binary_targets)
							ids = torch.cat(list_ids)

							if evaluate_representation:
								# We do not save results anyway. We only print those for test_annotator = train_annotator
								if test_annotator == train_annotator:
									all_representation_test_metrics = test_representations_various_probers(
										{'feats': representations, 'labels': binary_targets, 'ids': ids},
										type_probers=type_probers)
									print(f"Evaluation of representation on test set :")
									for type_prober, representation_test_metrics in zip(type_probers,
																						all_representation_test_metrics):
										print(f"Prober {type_prober} :"
											  f"\nAverage (across movie clusters) : {representation_test_metrics['average']}"
											  f"\nGlobal : {representation_test_metrics['global']}")

										cv_results_all[type_prober].append(representation_test_metrics)

							if evaluate_representation_classifier:
								probabilities = torch.sigmoid(logits) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.softmax(logits, dim=-1)
								binary_targets_np = binary_targets.detach().cpu().numpy()
								probabilities_np = probabilities.detach().cpu().numpy().round(3)

								# For binary classification, we may desire a threshold different from 0.5
								# In this case, the "optimal" threshold is defined according to f1_macro on validation set
								# We may also want the results of validation inference (for plotting)
								if ((variable_threshold and num_output_neurons == 1) or save_val_inference) and ("vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture):
									# N.B. : We first need to recompute validation output probabilities ;
									# indeed, if the model is already trained, we won't go through the training loop again
									# (We do it for ViT models only to not copy the whole computation code)
									list_probabilities = []
									list_binary_targets = []
									with ((torch.no_grad())):
										for i_val, vbatch in enumerate(val_loader):
											visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = vbatch
											clip_label, one_hot_concepts, ids = (clip_label.to(device), one_hot_concepts.to(device), ids.to(device))
											if multimodal:
												for i_mod in range(len(visual_features)):
													visual_features[i_mod] = visual_features[i_mod].to(device)
													visual_mask[i_mod] = visual_mask[i_mod].to(device)
											else:
												visual_features = visual_features.to(device)
												visual_mask = visual_mask.to(device)
											representations, logits, c_sem = standard_inference(
												representation_architecture=representation_architecture,
												vit_model=vit_model,
												visual_features=visual_features,
												visual_mask=visual_mask, ids=ids,
												clip_label=None if no_cheating_val else clip_label,
												sampling_rate=0.)
											v_probabilities = torch.sigmoid(logits)
											list_probabilities.append(v_probabilities)
											list_binary_targets.append(clip_label)
									val_binary_targets = torch.cat(list_binary_targets)
									val_probabilities = torch.cat(list_probabilities)
									# N.B. : Be careful not to overwrite probabilities, probabilities_np and binary_targts_np from the test set
									val_binary_targets_np = val_binary_targets.detach().cpu().numpy()
									val_probabilities_np = val_probabilities.detach().cpu().numpy()

									# Now we have the validation results, so we can test the various possible thresholds
									if variable_threshold and num_output_neurons == 1:
										thresholds_to_test = np.linspace(0.01,0.99,99)
										threshold = 0.
										best_f1_macro = 0.
										for t in thresholds_to_test:
											y_pred = (val_probabilities >= t).int()
											y_pred = y_pred.detach().cpu().numpy()
											f1_macro = f1_score(val_binary_targets_np, y_pred, average='macro')
											if f1_macro > best_f1_macro:
												best_f1_macro = f1_macro
												threshold = t
										print(f"Optimal threshold (according to validation macro-F1) : {threshold}")
									else:
										threshold = 0.5

									# If desired, we can save the validation results
									if save_val_inference:
										val_df = get_specific_df(gt_csv=gt_csv, classif_info=classif_info, folds_info=folds_info, fold_idx=fold_idx, annotator=test_annotator)
										val_df = get_val_inferred_df(val_df, val_probabilities_np)
										val_df.loc[:, "optimal_threshold"] = threshold if variable_threshold and num_output_neurons == 1 else np.nan
										csv_directory = f"{PROJECT_DIR}/results_two_stage/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{repr_str}_full_model/train_{train_annotator}"
										if not os.path.exists(csv_directory):
											os.makedirs(csv_directory)
										combined_csv_path = f"{csv_directory}/val_results_seed_{seed}_{test_annotator}_{exp_name}_fold_{fold_idx}{pseudo_labeling_str}.csv"
										val_df.to_csv(combined_csv_path, sep=';', index=False)
										print(f"\nSaved successfully val inference per-observation results at {combined_csv_path}")

								else:
									threshold = 0.5

								pred_label = (probabilities >= threshold).int() if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.argmax(probabilities, dim=-1).int()
								pred_label_np = pred_label.detach().cpu().numpy()

								test_classif_metrics = calculate_tb_metrics(np.nan, binary_targets_np, pred_label_np,
																			probabilities_np,
																		 num_output_neurons=num_output_neurons)

								# We only print the results corresponding to the train annotator, but save all
								if test_annotator == train_annotator:
									print(f"Evaluation of classification of the core model on test set :\n{test_classif_metrics}")
									cv_results_classif_core.append(test_classif_metrics)

								# For finer processing of results later, we also save the per-observation results, in a similar fashion to infer.py
								original_df = get_specific_df(gt_csv=gt_csv, classif_info=classif_info, folds_info=folds_info,
															  fold_idx=fold_idx, annotator=test_annotator)
								original_df = get_test_inferred_df(original_df, probabilities_np)
								original_df.loc[:, "optimal_threshold"] = threshold if variable_threshold and num_output_neurons==1 else np.nan

								csv_directory = f"{PROJECT_DIR}/results_two_stage/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{repr_str}_full_model/train_{train_annotator}"
								if not os.path.exists(csv_directory):
									os.makedirs(csv_directory)

								combined_csv_path = f"{csv_directory}/test_results_seed_{seed}_{test_annotator}_{exp_name}_fold_{fold_idx}{pseudo_labeling_str}.csv"
								original_df.to_csv(combined_csv_path, sep=';', index=False)
								print(f"\nSaved successfully test inference per-observation results at {combined_csv_path}")

								# We may also want to save train inference :
								# for more compact code, we do it only for ViT and MLP
								if save_train_inference and ("vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture):
									list_probabilities = []
									with ((torch.no_grad())):
										for i_val, vbatch in enumerate(unshuffled_train_loader): # N.B. : it is not defined for differential training
											visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = vbatch
											clip_label, one_hot_concepts, ids = (clip_label.to(device),
																				 one_hot_concepts.to(device),
																				 ids.to(device))
											if multimodal:
												for i_mod in range(len(visual_features)):
													visual_features[i_mod] = visual_features[i_mod].to(device)
													visual_mask[i_mod] = visual_mask[i_mod].to(device)
											else:
												visual_features = visual_features.to(device)
												visual_mask = visual_mask.to(device)
											representations, logits, c_sem = standard_inference(
												representation_architecture=representation_architecture,
												vit_model=vit_model,
												visual_features=visual_features,
												visual_mask=visual_mask, ids=ids,
												clip_label=None if no_cheating_val else clip_label,
												sampling_rate=0.)
											v_probabilities = torch.sigmoid(logits)
											list_probabilities.append(v_probabilities)
									train_probabilities = torch.cat(list_probabilities)
									train_probabilities_np = train_probabilities.detach().cpu().numpy()

									# Actual saving of results
									train_df = unbalanced_train_dataset.gt_df
									train_df = get_train_inferred_df(train_df, train_probabilities_np)
									train_df.loc[:, "optimal_threshold"] = threshold if variable_threshold and num_output_neurons == 1 else np.nan
									csv_directory = f"{PROJECT_DIR}/results_two_stage/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{repr_str}_full_model/train_{train_annotator}"
									if not os.path.exists(csv_directory):
										os.makedirs(csv_directory)
									combined_csv_path = f"{csv_directory}/train_results_seed_{seed}_{test_annotator}_{exp_name}_fold_{fold_idx}{pseudo_labeling_str}.csv"
									train_df.to_csv(combined_csv_path, sep=';', index=False)
									print(f"\nSaved successfully train inference per-observation results at {combined_csv_path}")

						# clear
						torch.cuda.empty_cache()

					# TEST ON THE UNLABELED SET (USED FOR COMPARING VARIOUS PSEUDO-MODELLING APPROACHES)
					if test_unlabeled_set:
						list_logits = []
						with ((torch.no_grad())):
							for i_val, tbatch in enumerate(unlabeled_loader):
								visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = tbatch

								clip_label, one_hot_concepts, ids = (
									clip_label.to(device),
									one_hot_concepts.to(device),
									ids.to(device))
								if multimodal:
									for i_mod in range(len(visual_features)):
										visual_features[i_mod] = visual_features[i_mod].to(device)
										visual_mask[i_mod] = visual_mask[i_mod].to(device)
								else:
									visual_features = visual_features.to(device)
									visual_mask = visual_mask.to(device)

								if "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
									representations, logits, c_sem = standard_inference(
										representation_architecture=representation_architecture,
										vit_model=vit_model,
										visual_features=visual_features,
										visual_mask=visual_mask, ids=ids,
										clip_label=None if no_cheating_test else clip_label,
										sampling_rate=0.)
								else:
									raise Exception(f"Unsupported representation_architecture for testing unlabeled set: {representation_architecture}")

								list_logits.append(logits)

						logits = torch.cat(list_logits)
						probabilities = torch.sigmoid(logits) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.softmax(logits, dim=-1)
						probabilities_np = probabilities.detach().cpu().numpy().round(3)

						# For finer processing of results later, we also save the per-observation results, in a similar fashion to infer.py
						original_df = get_specific_df(gt_csv=gt_csv, classif_info=classif_info, folds_info=folds_info, fold_idx=fold_idx, annotator=train_annotator)
						# In case of pseudo-labeling, for relevant testing, we replace the original labels by the pseudo-labels that were used at this training cycle
						if pseudo_labeling:
							classif_classes = classif_info["classif_classes"]
							if classif_info['type_classif'] == 'label':
								reverse_dico = {v: k for k, v in classif_classes.items()}
							else:
								raise Exception(f"classif_info type_classif {classif_info['type_classif']} not handled yet for pseudo-labeling")
							for guid, pseudo_label in zip(pseudo_labels_guids, pseudo_labels):
								original_df.loc[original_df['guid'] == guid, 'label'] = reverse_dico[pseudo_label]

						original_df = get_unlabeled_inferred_df(original_df, probabilities_np,
																pseudo_labeling_val_to_unlabeled=pseudo_labeling_val_to_unlabeled,
																folds_info=folds_info, fold_idx=fold_idx)
						original_df.loc[:, "optimal_threshold"] = np.nan

						csv_directory = f"{PROJECT_DIR}/results_two_stage/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{repr_str}_full_model/train_{train_annotator}"
						if not os.path.exists(csv_directory):
							os.makedirs(csv_directory)
						combined_csv_path = f"{csv_directory}/unlabeled_results_seed_{seed}_{train_annotator}_{exp_name}_fold_{fold_idx}{pseudo_labeling_str}.csv"
						original_df.to_csv(combined_csv_path, sep=';', index=False)
						print(f"\nSaved successfully per-observation results of unlabeled set at {combined_csv_path}")


					# CUSTOM TESTING (for now only implemented for full representation models)
					if test_on_custom_dataset is not None and representation_architecture not in ["xclipemb", "representation_vit"]:
						custom_dataset = FeatureDF(modality=modality, gt_csv=test_on_custom_dataset, split="test", fold_idx=fold_idx,
												   folds_info=folds_info, classif_info=classif_info, max_seq_len=max_seq_len,
												   seed=seed, annotator=custom_annotator, with_guid=True, with_id=True,
								  representation_architecture=representation_architecture,
												   custom_clips=custom_clips,
											  specified_visual_feature_dir=specified_visual_feature_dir)
						# As custom datasets are usually small and created using guids,
						# we do not care about restricting it to a given annotator
						if len(custom_dataset) > 0: # they are so small that they can be empty for some folds
							custom_loader = DataLoader(custom_dataset, batch_size=len(custom_dataset) if "differential" in representation_architecture else batch_size, shuffle=False, num_workers=19,
													 collate_fn=my_collate_fn)
							list_probabilities = []

							with ((torch.no_grad())):
								for i_val, tbatch in enumerate(custom_loader):
									# N.B. : Note that, as test_loader is implemented for now,
									# the test dataset actually corresponds to only one batch
									visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = tbatch
									clip_label, one_hot_concepts, ids = (
										clip_label.to(device),
										one_hot_concepts.to(device),
										ids.to(device))
									if multimodal:
										for i_mod in range(len(visual_features)):
											visual_features[i_mod] = visual_features[i_mod].to(device)
											visual_mask[i_mod] = visual_mask[i_mod].to(device)
									else:
										visual_features = visual_features.to(device)
										visual_mask = visual_mask.to(device)
									binary_targets = clip_label.float().to(device) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else clip_label.type(torch.LongTensor).to(device)
									# N.B. : Somehow changing type with type() may change the device, so we need to specify .to(device) again
									binary_targets_for_loss = binary_targets.unsqueeze(1) if num_output_neurons == 1 else binary_targets

									if "MIL" in representation_loss:
										test_batch_size = len(visual_features)
										# flattening is necessary for the model to work well
										# For code generalization it is better to do flattening outside of the model
										if "mlp" in representation_architecture:
											visual_features = visual_features.view(test_batch_size * nb_segment,
																				   feature_dim)
										elif "vit" in representation_architecture:
											visual_features = visual_features.view(test_batch_size * nb_segment,
																				   max_seq_len, feature_dim)
											visual_mask = visual_mask.view(test_batch_size * nb_segment, max_seq_len)
										else:
											raise Exception(f"Unknown representation_architecture : '{representation_architecture}'")

									# Now we do the inference
									if representation_architecture == "representation_vit":
										logits = torch.zeros(1)  # representation_vit returns no logits
										representations = vit_model(visual_features, visual_mask)
									elif "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
										representations, logits, c_sem = standard_inference(
											representation_architecture=representation_architecture,
											vit_model=vit_model,
											visual_features=visual_features,
											visual_mask=visual_mask, ids=ids,
											clip_label=None if no_cheating_test else clip_label,
											sampling_rate=0.)
									elif "dann" in representation_architecture:
										representations, logits, domain_logits = dann_inference(
											representation_architecture=representation_architecture,
											vit_model=vit_model, visual_features=visual_features,
											visual_mask=visual_mask, ids=ids,
											clip_label=None if no_cheating_test else clip_label)
									elif representation_architecture == "xclipemb":
										logits = torch.zeros(1)  # xclipemb returns no logits
										representations = torch.mean(visual_features, dim=1)
									else:
										raise Exception(
											f"Unsupported representation_architecture : '{representation_architecture}'")

									# If MIL model, we reshape the results
									if "MIL" in representation_loss:
										logits = logits.view(test_batch_size, nb_segment, 1)
										logits, _ = torch.max(logits, dim=1)  # shape [batch_size, 1]
										# Similarly for representations
										representations = representations.view(test_batch_size, nb_segment, -1)
										representations = torch.mean(representations, dim=1)  # shape [batch_size, repr_dim]

									probabilities = torch.sigmoid(logits) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.softmax(logits, dim=-1)
									list_probabilities.append(probabilities)

							probabilities = torch.cat(list_probabilities)
							probabilities_np = probabilities.detach().cpu().numpy().round(3)

							# For finer processing of results later, we also save the per-observation results, in a similar fashion to infer.py
							original_df = get_specific_df(gt_csv=test_on_custom_dataset, classif_info=classif_info, folds_info=folds_info,
															  fold_idx=fold_idx, annotator=custom_annotator)
							original_df = get_test_inferred_df(original_df, probabilities_np)

							csv_directory = f"{PROJECT_DIR}/results_two_stage/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{repr_str}_full_model/train_{train_annotator}"
							if not os.path.exists(csv_directory):
								os.makedirs(csv_directory)

							name_custom_dataset = (test_on_custom_dataset.split('/')[-1]).split('.')[0]
							combined_csv_path = f"{csv_directory}/{name_custom_dataset}_results_seed_{seed}_{custom_annotator}_{exp_name}_fold_{fold_idx}{pseudo_labeling_str}.csv"
							original_df.to_csv(combined_csv_path, sep=';', index=False)
							print(f"\nSaved successfully per-observation results at {combined_csv_path}")

					# We have trained the embedding model for this seed and annotator
					# For the classification, either we train a classifier or, because our embedding space is supposed
					# to have embedding properties, we can use non-parametric (and thus non-trainable) methods :
					# k-NN or its more complex version based on similarity.
					# These non-parametric methods require "few-shot samples"
					# Note that, because it is a derivable expression, we can also train the embeddings using the CE-loss

					# SUCCESSIVE TRAINING OF CLASSIFIER
					train_classif = True
					if not consider_classif:
						train_classif = False
						print(f"\nNO TRAIN OF CLASSIFICATION : WE ARE NOT INTERESTED IN IT FOR NOW")
					else:
						if use_trained_classification_models and existing_classification_model:
							train_classif = False
							print(f"\nNO TRAIN OF CLASSIFICATION : WE USE THE ALREADY TRAINED MODEL SAVED AT "
								  f"{global_model_path_classification}")
						if "similarity" in type_classif:
							train_classif = False
							print(f"\nNO TRAIN OF CLASSIFICATION : {type_classif} IS NON-PARAMETRIC")
						if type_classif in ['logistic', 'knn', 'prior', 'uniform']:
							train_classif = False  # the training of these models doesn't need to be hand-designed

					if train_classif:
						print(f"\nNOW TRAINING MLP CLASSIFICATION...")
						# We use this part of code for MLP classifier only
						classification_head = ClassificationHead(dim=512 if representation_architecture == "xclipemb" else dim,
																 dropout=dropout, num_classes=num_output_neurons).to(device)
						optimizer_class = optim.Adam(classification_head.parameters(), lr=learning_rate_class,
													 weight_decay=0.00005)
						scheduler_class = ReduceLROnPlateau(optimizer_class, 'min', patience=5, factor=0.1)

						# CLASSIFICATION TRAINING
						epoch = 0
						global_step = 0
						best_vloss = 1_000_000.  # this variable will be used for checkpointing the best model (based on validation loss)
						impatience = 0
						val_losses = []
						early_stopped = False

						while not early_stopped and epoch < max_epochs_class:
							print(f"\nEPOCH {epoch} :")

							# Make sure gradient tracking is on
							classification_head.train(True)
							running_loss_class = 0.

							for i_train, batch in enumerate(train_loader_class):
								visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = batch
								clip_label, one_hot_concepts, ids = (
									clip_label.to(device),
									one_hot_concepts.to(device),
									ids.to(device))
								if multimodal:
									for i_mod in range(len(visual_features)):
										visual_features[i_mod] = visual_features[i_mod].to(device)
										visual_mask[i_mod] = visual_mask[i_mod].to(device)
								else:
									visual_features = visual_features.to(device)
									visual_mask = visual_mask.to(device)

								binary_targets = clip_label.float().to(device) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else clip_label.type(torch.LongTensor).to(device)
								# N.B. : Somehow changing type with type() may change the device, so we need to specify .to(device) again
								binary_targets_for_loss = binary_targets.unsqueeze(1) if num_output_neurons == 1 else binary_targets

								if representation_architecture == "representation_vit":
									representations = vit_model(visual_features, visual_mask)
								elif "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
									representations, logits, c_sem = standard_inference(
										representation_architecture=representation_architecture,
										vit_model=vit_model,
										visual_features=visual_features,
										visual_mask=visual_mask, ids=ids,
										clip_label=clip_label,
										sampling_rate=0.)
								elif "dann" in representation_architecture:
									representations, classif_logits, domain_logits = dann_inference(representation_architecture=representation_architecture,
																		   vit_model=vit_model, visual_features=visual_features,
																		   visual_mask=visual_mask, ids=ids, clip_label=clip_label)
								elif representation_architecture == "xclipemb":
									pass
								else:
									raise Exception(
										f"Unsupported representation_architecture : '{representation_architecture}'")
								# N.B. : We do nothing special with MIL-trained representation models

								logits = classification_head(representations)
								loss_class = loss_ce(logits, binary_targets_for_loss)
								optimizer_class.zero_grad()  # Zero your gradients for every batch!
								loss_class.backward(retain_graph=True)
								# Adjust learning weights
								optimizer_class.step()

								# We compute related variables that can be used for computing metrics
								# See calculate_tb_metrics method of the Lightning module if I want to compute them
								probabilities = torch.sigmoid(logits) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.softmax(logits, dim=-1)
								pred_label = (probabilities >= 0.5).int() if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.argmax(probabilities, dim=-1).int()
								binary_targets_np = binary_targets.detach().cpu().numpy()
								pred_label_np = pred_label.detach().cpu().numpy()
								probabilities_np = probabilities.detach().cpu().numpy()
								# Tensorboard logging : for training, it is every step/batch
								train_metrics = calculate_tb_metrics(loss_class, binary_targets_np, pred_label_np,
																	 probabilities_np,
																		 num_output_neurons=num_output_neurons)
								for k, v in train_metrics.items():
									writer.add_scalar(f"{k}/train", v, global_step=epoch)

								warm_up_lr_adaptation(global_step, warmup_steps_class, learning_rate_class, optimizer_class)

								# Gather data and report
								running_loss_class += loss_class.item()

								global_step += 1

							avg_loss_class = running_loss_class / (i_train + 1)
							print(f"Classification train loss : {avg_loss_class}")

							# VALIDATION
							running_vloss_class = 0.0
							classification_head.eval()
							# Set the model to evaluation mode, disabling dropout and using population
							# statistics for batch normalization.

							# Disable gradient computation and reduce memory consumption.
							with torch.no_grad():
								list_probabilities = []
								list_binary_targets = []
								list_pred_label = []
								for i_val, vbatch in enumerate(val_loader):
									# N.B. : Note that, as it val_loader is implemented for now,
									# the validation dataset actually corresponds to only one batch
									visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = vbatch
									clip_label, one_hot_concepts, ids = (
										clip_label.to(device),
										one_hot_concepts.to(device),
										ids.to(device))
									if multimodal:
										for i_mod in range(len(visual_features)):
											visual_features[i_mod] = visual_features[i_mod].to(device)
											visual_mask[i_mod] = visual_mask[i_mod].to(device)
									else:
										visual_features = visual_features.to(device)
										visual_mask = visual_mask.to(device)
									binary_targets = clip_label.float().to(device) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else clip_label.type(torch.LongTensor).to(device)
									# N.B. : Somehow changing type with type() may change the device, so we need to specify .to(device) again
									binary_targets_for_loss = binary_targets.unsqueeze(1) if num_output_neurons == 1 else binary_targets

									if representation_architecture == "representation_vit":
										representations = vit_model(visual_features, visual_mask)
									elif "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
										representations, logits, c_sem = standard_inference(
											representation_architecture=representation_architecture,
											vit_model=vit_model,
											visual_features=visual_features,
											visual_mask=visual_mask, ids=ids,
											clip_label=None if no_cheating_val else clip_label,
											sampling_rate=0.)
									elif "dann" in representation_architecture:
										representations, classif_logits, domain_logits = dann_inference(
											representation_architecture=representation_architecture,
											vit_model=vit_model, visual_features=visual_features,
											visual_mask=visual_mask, ids=ids, clip_label=None if no_cheating_val else clip_label)
									elif representation_architecture == "xclipemb":
										representations = torch.mean(visual_features, dim=1)
									else:
										raise Exception(
											f"Unsupported representation_architecture : '{representation_architecture}'")
									# N.B. : We do nothing special with MIL-trained representation models

									logits = classification_head(representations)
									vloss_class = loss_ce_val(logits, binary_targets_for_loss)
									running_vloss_class += vloss_class.item()

									# We compute related variables that can be used for computing metrics
									# See calculate_tb_metrics method of the Lightning module if I want to compute them
									probabilities = torch.sigmoid(logits) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.softmax(logits, dim=-1)
									pred_label = (probabilities >= 0.5).int() if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.argmax(probabilities, dim=-1).int()

									list_probabilities.append(probabilities)
									list_pred_label.append(pred_label)
									list_binary_targets.append(clip_label)

							avg_vloss_class = running_vloss_class / (i_val + 1)
							print(f"Classification validation loss : {avg_vloss_class}")

							probabilities = torch.cat(list_probabilities)
							pred_label = torch.cat(list_pred_label)
							binary_targets = torch.cat(list_binary_targets)
							pred_label_np = pred_label.detach().cpu().numpy()
							probabilities_np = probabilities.detach().cpu().numpy()
							binary_targets_np = binary_targets.detach().cpu().numpy()

							# Tensorboard logging : for validation, it is every epoch
							val_metrics = calculate_tb_metrics(avg_vloss_class, binary_targets_np, pred_label_np,
															   probabilities_np,
																		 num_output_neurons=num_output_neurons)
							for k, v in val_metrics.items():
								writer.add_scalar(f"{k}/val", v, global_step=epoch)
							# Print logging
							print(f"Validation metrics : {val_metrics}")

							val_losses.append(avg_vloss_class)

							# Tracks best performance, and save the model's state
							if avg_vloss_class < best_vloss:
								print(f"New best model (epoch {epoch})")
								best_vloss = avg_vloss_class
								impatience = 0
								if save_model:
									torch.save(classification_head.state_dict(), global_model_path_classification)
							else:
								impatience += 1

							early_stopped = (epoch + 1 >= min_epochs_class) and (impatience >= patience_class)
							if early_stopped:
								print(f"Early stopping\nSuccessive val losses : {[round(vl, 2) for vl in val_losses]}")

							scheduler_class.step(avg_vloss_class)
							epoch += 1

						print(f"\nSUCCESSFUL TRAINING OF CLASSIFICATION")
						# clear
						del classification_head
						torch.cuda.empty_cache()

					# VALIDATION-DETERMINED BETA AND K PARAMETERS FOR SIMILARITY-BASED CLASSIFIER
					if consider_classif and "similarity" in type_classif:
						with_clustering = type_classif == "similarity_clustering"
						# Even if we do the clustering variation, at first we only focus on beta
						if len(similarity_betas) > 1:
							print(f"DETERMINING OPTIMAL BETA PARAMETER ON VALIDATION SET")
							beta_metrics = dict()

							# We define the similarity-based classification model
							# We have to retrieve all training embeddings, labels and ids from the train loader
							train_embeddings, train_labels, train_ids = (
								get_embeddings_labels_ids_from_loader(loader=train_loader_class, device=device,
											  representation_architecture=representation_architecture, vit_model=vit_model))

							classification_head = SimilarityBasedClassifier(train_embeddings,
																			train_labels,
																			train_ids,
																			beta=similarity_betas[0],
																			# this value will be changed anyway
																			device=device,
																			similarity_type=type_classif,
																			clustering_k=similarity_clustering_ks[
																				0] if with_clustering else None)

							for beta in similarity_betas:
								# We test beta on the validation set
								classification_head.set_beta(beta=beta)
								representations, val_labels, val_ids = (
									get_embeddings_labels_ids_from_loader(loader=val_loader, device=device,
																		  representation_architecture=representation_architecture,
																		  vit_model=vit_model))
								logits = classification_head(representations)
								probabilities = torch.softmax(logits, dim=1)[:, 1]
								# in their definition and use in articles, similarity logits are meaningful from
								# the point of view of accuracy, ie. for argmax prediction, but we should not
								# expect any good probabilistic properties
								binary_targets = val_labels.float if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else clip_label.type(torch.LongTensor)
								pred_label = torch.argmax(probabilities).int()
								binary_targets_np = binary_targets.detach().cpu().numpy()
								pred_label_np = pred_label.detach().cpu().numpy()
								probabilities_np = probabilities.detach().cpu().numpy()

								val_classif_metrics = calculate_tb_metrics(np.nan, binary_targets_np, pred_label_np,
																		   probabilities_np,
																		 num_output_neurons=num_output_neurons)
								# Because there may not be a loss associated to our classification head, we indicate np.nan as loss
								print(f"Eval of classification on validation set for beta={beta} :\n{val_classif_metrics}")
								beta_metrics[beta] = val_classif_metrics['f1_w']
							# f1_w seems like the best single measure of performance

							# clear
							del classification_head
							torch.cuda.empty_cache()

							final_beta = max(beta_metrics, key=beta_metrics.get)

						else:
							final_beta = similarity_betas[0]

						print(f"\nFINAL BETA (for this fold) is {final_beta}")

						# Now, if we do the clustering variation, we look for the optimal k
						if with_clustering and len(similarity_clustering_ks) > 1:
							print(f"\nDETERMINING OPTIMAL K PARAMETER ON VALIDATION SET")
							k_metrics = dict()

							# We define the similarity-based classification model
							# We have to retrieve all training embeddings, labels and ids from the train loader
							train_embeddings, train_labels, train_ids = (
								get_embeddings_labels_ids_from_loader(loader=train_loader_class, device=device,
																	  representation_architecture=representation_architecture,
																	  vit_model=vit_model))

							classification_head = SimilarityBasedClassifier(train_embeddings,
																			train_labels,
																			train_ids,
																			beta=final_beta,
																			device=device,
																			similarity_type=type_classif,
																			clustering_k=similarity_clustering_ks[0]
																			# this value will be changed anyway
																			)

							for k in similarity_clustering_ks:
								# We test beta on the validation set
								classification_head.set_clustering_k(k=k)
								representations, val_labels, val_ids = (
									get_embeddings_labels_ids_from_loader(loader=val_loader, device=device,
																		  representation_architecture=representation_architecture,
																		  vit_model=vit_model))
								logits = classification_head(representations)
								probabilities = torch.softmax(logits, dim=1)[:, 1]
								# in their definition and use in articles, similarity logits are meaningful from
								# the point of view of accuracy, ie. for argmax prediction, but we should not
								# expect any good probabilistic properties
								binary_targets = val_labels.float if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else clip_label.type(torch.LongTensor)
								pred_label = torch.argmax(probabilities).int()
								binary_targets_np = binary_targets.detach().cpu().numpy()
								pred_label_np = pred_label.detach().cpu().numpy()
								probabilities_np = probabilities.detach().cpu().numpy()

								val_classif_metrics = calculate_tb_metrics(np.nan, binary_targets_np, pred_label_np,
																		   probabilities_np,
																		 num_output_neurons=num_output_neurons)
								# Because there may not be a loss associated to our classification head, we indicate np.nan as loss
								print(f"Evaluation of classification on validation set for k={k} :\n{val_classif_metrics}")
								k_metrics[k] = val_classif_metrics['f1_w']
							# f1_w seems like the best single measure of performance

							# clear
							del classification_head
							torch.cuda.empty_cache()

							final_k = max(k_metrics, key=k_metrics.get)

						else:
							final_k = similarity_clustering_ks[0] if with_clustering else None

						if with_clustering:
							print(f"\nFINAL K (for this fold) is {final_k}")

					# EVALUATION OF CLASSIFICATION ON THE TEST SET (on the best model, not the last !) FOR THIS FOLD
					if consider_classif:

						# We load the classification model
						if type_classif == "mlp":
							classification_head = ClassificationHead(dim=512 if representation_architecture == "xclipemb" else dim,
																	 dropout=dropout, num_classes=num_output_neurons).to(device)
							classification_head.load_state_dict(
								torch.load(global_model_path_classification, map_location=device,
										   weights_only=True))
							classification_head.eval()
						# We "train" the similarity and sklearn models here
						elif "similarity" in type_classif or type_classif in ['logistic', 'knn', 'prior', 'uniform']:
							# We have to retrieve all training embeddings, labels and ids from the train loader
							train_embeddings, train_labels, train_ids = (
								get_embeddings_labels_ids_from_loader(loader=train_loader_class, device=device,
																	  representation_architecture=representation_architecture,
																	  vit_model=vit_model))

							if type_classif in ['logistic', 'knn', 'prior', 'uniform']:
								print(f"\nNOW TRAINING {type_classif} classification")
								classification_head, scoring_clf = get_linear_clf(type_classif)
								classification_head.fit(train_embeddings.detach().cpu().numpy(),
														train_labels.detach().cpu().numpy())

							else:
								classification_head = SimilarityBasedClassifier(train_embeddings,
																				train_labels,
																				train_ids,
																				beta=final_beta,
																				device=device,
																				similarity_type=type_classif,
																				clustering_k=final_k)

						for test_annotator in test_annotators:
							test_dataset = FeatureDF(modality=modality, gt_csv=gt_csv, split="test", fold_idx=fold_idx,
													 folds_info=folds_info, classif_info=classif_info,
													 max_seq_len=max_seq_len,
													 seed=seed, annotator=test_annotator, with_guid=True, with_id=True,
								  representation_architecture=representation_architecture,
											  specified_visual_feature_dir=specified_visual_feature_dir)
							test_loader = DataLoader(test_dataset, batch_size=len(test_dataset) if "differential" in representation_architecture else batch_size,
													 shuffle=False, num_workers=19, collate_fn=my_collate_fn)
							# Thanks to shuffle=False, we will be able to store test probabilities directly (without having to recompute the inference order)

							with ((torch.no_grad())):
								list_binary_targets = []
								list_probabilities = []
								for i_val, tbatch in enumerate(test_loader):
									# N.B. : Note that, as test_loader is implemented for now,
									# the test dataset actually corresponds to only one batch
									visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = tbatch
									clip_label, one_hot_concepts, ids = (
										clip_label.to(device),
										one_hot_concepts.to(device),
										ids.to(device))
									if multimodal:
										for i_mod in range(len(visual_features)):
											visual_features[i_mod] = visual_features[i_mod].to(device)
											visual_mask[i_mod] = visual_mask[i_mod].to(device)
									else:
										visual_features = visual_features.to(device)
										visual_mask = visual_mask.to(device)
									binary_targets = clip_label.float().to(device) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else clip_label.type(torch.LongTensor).to(device)
									# N.B. : Somehow changing type with type() may change the device, so we need to specify .to(device) again
									binary_targets_for_loss = binary_targets.unsqueeze(1) if num_output_neurons == 1 else binary_targets

									# Representation
									if representation_architecture == "representation_vit":
										representations = vit_model(visual_features, visual_mask)
									elif "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
										representations, logits, c_sem = standard_inference(
											representation_architecture=representation_architecture,
											vit_model=vit_model,
											visual_features=visual_features,
											visual_mask=visual_mask, ids=ids,
											clip_label=None if no_cheating_test else clip_label,
											sampling_rate=0.)
									elif "dann" in representation_architecture:
										representations, classif_logits, domain_logits = dann_inference(
											representation_architecture=representation_architecture,
											vit_model=vit_model, visual_features=visual_features,
											visual_mask=visual_mask, ids=ids, clip_label=None if no_cheating_test else clip_label)
									elif representation_architecture == "xclipemb":
										representations = torch.mean(visual_features, dim=1)
									else:
										raise Exception(
											f"Unsupported representation_architecture : '{representation_architecture}'")
									# N.B. : We do nothing special with MIL-trained representation models

									# Classification
									if type_classif == "mlp":
										logits = classification_head(representations)
										probabilities = torch.sigmoid(logits)
									elif "similarity" in type_classif:
										logits = classification_head(representations)
										probabilities = torch.softmax(logits, dim=1)[:, 1]
									# in their definition and use in articles, similarity logits are meaningful from
									# the point of view of accuracy, ie. for argmax prediction, but we should not
									# expect any good probabilistic properties
									elif type_classif in ['logistic', 'knn', 'prior', 'uniform']:
										probabilities = classification_head.predict_proba(representations.detach().cpu().numpy())[:, 1]
										# For generalization of the code below, we convert it
										probabilities = torch.from_numpy(probabilities).float().to(device)

									list_binary_targets.append(binary_targets)
									list_probabilities.append(probabilities)

							probabilities = torch.cat(list_probabilities)
							binary_targets = torch.cat(list_binary_targets)
							pred_label = (probabilities >= 0.5).int() if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.argmax(probabilities, dim=-1).int()

							binary_targets_np = binary_targets.detach().cpu().numpy()
							pred_label_np = pred_label.detach().cpu().numpy()
							probabilities_np = probabilities.detach().cpu().numpy().round(3)

							test_classif_metrics = calculate_tb_metrics(np.nan, binary_targets_np, pred_label_np,
																		probabilities_np,
																		 num_output_neurons=num_output_neurons)
							# We only print the results for test_annotator == train_annotator, although we save all
							if test_annotator == train_annotator:
								print(f"Evaluation of classification on test set :\n{test_classif_metrics}")
								cv_results_classif.append(test_classif_metrics)

							# For finer processing of results later, we also save the per-observation results, in a similar fashion to infer.py
							original_df = get_specific_df(gt_csv=gt_csv, classif_info=classif_info, folds_info=folds_info,
														  fold_idx=fold_idx, annotator=test_annotator)
							original_df = get_test_inferred_df(original_df, probabilities_np)

							csv_directory = f"{PROJECT_DIR}/results_two_stage/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{type_classif_str}_{repr_str}/train_{train_annotator}"
							if not os.path.exists(csv_directory):
								os.makedirs(csv_directory)

							combined_csv_path = f"{csv_directory}/test_results_seed_{seed}_{test_annotator}_{exp_name}_fold_{fold_idx}{pseudo_labeling_str}.csv"
							original_df.to_csv(combined_csv_path, sep=';', index=False)
							print(f"\nSaved successfully per-observation results at {combined_csv_path}")

						# clear
						del classification_head
						torch.cuda.empty_cache()

					if pseudo_labeling:
						training_cycles_finished = (pseudo_label_proportion >= max_pseudo_label_proportion) # or (remaining_budget <= 0) # we do not want to stop once
						# We get ready for the next training cycle
						if not training_cycles_finished:
							training_cycle_id += 1
							pseudo_label_proportion += stepping_threshold
							# We do the inference of the trained model on the unlabeled dataset :
							full_logits_np = []
							full_guids_np = []
							full_binary_targets_np = [] # only used for verbose purpose, to check which pseudo-labels are different from original labels
							for i_u, batch in enumerate(unlabeled_loader):
								# a) we load the batch
								visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = batch
								clip_label, one_hot_concepts, ids = (clip_label.to(device), one_hot_concepts.to(device), ids.to(device))
								if multimodal:
									for i_mod in range(len(visual_features)):
										visual_features[i_mod] = visual_features[i_mod].to(device)
										visual_mask[i_mod] = visual_mask[i_mod].to(device)
								else:
									visual_features = visual_features.to(device)
									visual_mask = visual_mask.to(device)
								binary_targets = clip_label.float().to(device) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else clip_label.type(torch.LongTensor).to(device)
								binary_targets_np = binary_targets.detach().cpu().numpy()
								# b) we do the proper inference (N.B.: for denser code, for now it is implemented for ViT/MLP only)
								if "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
									representations, logits, c_sem = standard_inference(
										representation_architecture=representation_architecture,
										vit_model=vit_model, visual_features=visual_features,
										visual_mask=visual_mask, ids=ids, clip_label=clip_label,
										)
								else:
									raise Exception(f'For now pseudo-labeling is implemented vor ViT and MLP models only')
								logits_np = logits.detach().cpu().numpy()
								# We format the logits so there are as many columns as classes
								if num_output_neurons == 1:
									negative_logits_np = np.zeros(logits_np.shape)
									unlabeled_logits_np = np.hstack((negative_logits_np, logits_np))
								else:
									unlabeled_logits_np = logits_np
								# Now we add those metrics to the list over batches
								full_logits_np.append(unlabeled_logits_np)
								full_guids_np.append(guids)
								full_binary_targets_np.append(binary_targets_np)

							full_binary_targets_np = np.concatenate(full_binary_targets_np).astype(np.int_)
							full_logits_np = np.concatenate(full_logits_np)
							full_guids_np = np.concatenate(full_guids_np)

							# In a "sampling without replacement" context (unlike original curriculum labeling code),
							# we need to restrict the unlabeled set to the examples that are not used yet as pseudo-labels
							already_used_guids = np.isin(full_guids_np, pseudo_labels_guids)
							not_used_guids = np.bitwise_not(already_used_guids)
							full_binary_targets_np = full_binary_targets_np[not_used_guids]
							full_logits_np = full_logits_np[not_used_guids]
							full_guids_np = full_guids_np[not_used_guids]

							# Now we have the inference logits and the corresponding guids of the unlabeled dataset
							# From there we first compute probabilities and pseudo-labels
							full_probabilities_np = softmax(full_logits_np, axis=1)
							full_pseudo_labels_np = np.argmax(full_probabilities_np, axis=1)

							# At the last training cycle, there may not be exactly pseudo_label_number remaining
							pseudo_label_number = min(pseudo_label_number, len(full_probabilities_np))

							if confidence_based_pseudo_labeling:
								full_confidences_np = full_probabilities_np[np.arange(len(full_probabilities_np)), full_binary_targets_np]
								# Now we select the pseudo_label_proportion least confident labels
								confidences_order = np.argsort(full_confidences_np)
								kept_indices = confidences_order[:pseudo_label_number]
								confidence_threshold = max(full_confidences_np[kept_indices])

							else: # pseudo-confidence-based pseudo_labeling = original curriculum labeling
								full_pseudo_confidences_np = full_probabilities_np[np.arange(len(full_probabilities_np)), full_pseudo_labels_np]
								# Now we select the pseudo_label_proportion most confident pseudo-labels
								pseudo_confidences_order = np.argsort(full_pseudo_confidences_np)
								kept_indices = pseudo_confidences_order[(-pseudo_label_number):]
								confidence_threshold = min(full_pseudo_confidences_np[kept_indices])

							# We add a verbosity step to better understand pseudo-labeling
							print(f"The (pseudo-)confidence threshold for keeping {pseudo_label_number} new pseudo-labels"
								f" (={round(stepping_threshold, 2)} of original unlabeled set) is {confidence_threshold}")

							n_pseudo_labels_guids = full_guids_np[kept_indices]
							n_pseudo_labels = full_pseudo_labels_np[kept_indices]
							n_pseudo_labels_original_labels = full_binary_targets_np[kept_indices]

							# We adapt the n_pseudo_labels depending on the remaining_budget
							# If there is no budget remaining, we still add the stepping_threshold proportion, but without changing any label
							if remaining_budget <=0:
								print(f"There is no remaining budget : we still add stepping_threshold proportion to X_train")
								pseudo_labels_guids = np.concatenate((pseudo_labels_guids, n_pseudo_labels_guids))
								pseudo_labels = np.concatenate((pseudo_labels, n_pseudo_labels_original_labels))
							else:
								# "new" = different from original label
								new_pseudo_labels_mask = n_pseudo_labels != n_pseudo_labels_original_labels
								# If we exceed the remaining budget, we restrict the set of "new" pseudo-labels,
								# keeping only the most pseudo-confident
								if np.count_nonzero(new_pseudo_labels_mask) > remaining_budget:
									true_idx = np.flatnonzero(new_pseudo_labels_mask)  # the True indices, ie. the indices that will cost some budget
									# We set to False all but the remaining_budget last changes (ie. all but the the most pseudo-confident)
									new_pseudo_labels_mask[true_idx[:(-remaining_budget)]] = False
									# Note that the "unkept pseudo labels" observations are not thrown out,
									# we just keep the original labels instead
									# We update the n_pseudo_labels :
									n_pseudo_labels = np.where(new_pseudo_labels_mask, n_pseudo_labels, n_pseudo_labels_original_labels)

								# Now we have the correct update, we recompute new_pseudo_label_mask for verbosity
								new_pseudo_labels_mask = n_pseudo_labels != n_pseudo_labels_original_labels
								remaining_budget -= np.count_nonzero(new_pseudo_labels_mask)
								# We update the all_different arrays
								all_different_pseudo_labels_guids = np.concatenate((all_different_pseudo_labels_guids, n_pseudo_labels_guids[new_pseudo_labels_mask]))
								all_different_pseudo_labels = np.concatenate((all_different_pseudo_labels, n_pseudo_labels[new_pseudo_labels_mask]))

								# Verbosity
								new_pseudo_labels_guids = n_pseudo_labels_guids[new_pseudo_labels_mask]
								prop_changed = round(len(new_pseudo_labels_guids) / len(n_pseudo_labels_guids), 2)
								print(f"Among the new kept pseudo-labels, {len(new_pseudo_labels_guids)} ({prop_changed} of) "
									  f"pseudo-labels are different from original labeling ; their guids are :\n{new_pseudo_labels_guids}")
								#print(f"The proportion of positive in the new selected pseudo-labels is "
								#	  f"{round(np.count_nonzero(n_pseudo_labels) / len(n_pseudo_labels), 2)}")
								#print(f"(The proportion of positive in the original labels of the selected pseudo-labels was "
								#	  f"{round(np.count_nonzero(n_pseudo_labels_original_labels) / len(n_pseudo_labels_original_labels), 2)})")
								interesting_df = unlabeled_dataset.gt_df[unlabeled_dataset.gt_df['guid'].isin(new_pseudo_labels_guids)]
								interesting_df = interesting_df.drop(columns = ['comment', 'subtitle', 'start_clip', 'end_clip'])
								print(f"interesting_df.to_string() :\n{interesting_df.head(10).to_string()}")

								print(f"The number of new observations we include in the train set is {len(n_pseudo_labels_guids)}")
								# should always be equal to pseudo-label number
								pseudo_labels_guids = np.concatenate((pseudo_labels_guids, n_pseudo_labels_guids))
								pseudo_labels = np.concatenate((pseudo_labels, n_pseudo_labels))

						else:
							# It means we have reached the end of training cycles ; in particular, we have relabeled
							# everything our budget allowed, so now we can save the relabeled unlabeled set
							original_df = get_specific_df(gt_csv=gt_csv, classif_info=classif_info,
														  folds_info=folds_info,
														  fold_idx=fold_idx, annotator=train_annotator)

							# Either we were working on the full dataset based on the 'labeled' column,
							# or in a cross-validation manner if pseudo_labeling_val_to_unlabeled

							labeled_columns = [name for name in original_df.columns if 'labeled' in name]
							labeled_column = 'labeled' if len(labeled_columns) == 1 else f'labeled_{fold_idx}'

							if pseudo_labeling_val_to_unlabeled:
								# We set to labeled/unlabeled the observations according to whether or not they belong to the "val set"
								original_df.loc[
									original_df.imdb_key.isin(folds_info[fold_idx]["validation"]), labeled_column] = 0
								original_df.loc[
									~original_df.imdb_key.isin(folds_info[fold_idx]["validation"]), labeled_column] = 1
							# In both cases, the unlabeled set now corresponds to the observations 'labeled'=0
							original_df = original_df.query(f'{labeled_column} == 0')

							# We replace the original labels by the pseudo-labels that were used at this training cycle
							# Note that, in EN_HNS exp for instance, all pseudo-labels HNS are set to 'Sure'
							classif_classes = classif_info["classif_classes"]
							if classif_info['type_classif'] == 'label':
								reverse_dico = {v: k for k, v in classif_classes.items()}
							else:
								raise Exception(
									f"classif_info type_classif {classif_info['type_classif']} not handled yet for pseudo-labeling")
							for guid, pseudo_label in zip(all_different_pseudo_labels_guids, all_different_pseudo_labels):
								original_df.loc[original_df['guid'] == guid, 'label'] = reverse_dico[pseudo_label]

							# We save the relabeled originally unlabeled dataset
							csv_directory = f"{DATASETS_DIR}/relabeled_datasets/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{repr_str}_full_model/train_{train_annotator}"
							if not os.path.exists(csv_directory):
								os.makedirs(csv_directory)
							combined_csv_path = f"{csv_directory}/{dataset_name}_{train_annotator}_relabeled_seed_{seed}_fold_{fold_idx}.csv"
							original_df.to_csv(combined_csv_path, sep=';', index=False)
							print(f"\nSaved successfully the relabelled set (for this fold and annotator) at {combined_csv_path}")

							# Separately, we save the relabeled val set
							if also_relabel_val_set is not None:
								print(f"\nAlso relabeling val set...")
								# We do the inference on the validation set (to compute pseudo-labels)

								# We do the test inference on the val set. Note that, for now, only ViT and MLP models are supported
								list_logits = []
								all_guids_np = []
								list_binary_targets = []
								with ((torch.no_grad())):
									for i_val, tbatch in enumerate(val_loader):
										visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = tbatch
										clip_label, one_hot_concepts, ids = (
											clip_label.to(device),
											one_hot_concepts.to(device),
											ids.to(device))
										if multimodal:
											for i_mod in range(len(visual_features)):
												visual_features[i_mod] = visual_features[i_mod].to(device)
												visual_mask[i_mod] = visual_mask[i_mod].to(device)
										else:
											visual_features = visual_features.to(device)
											visual_mask = visual_mask.to(device)

										binary_targets = clip_label.float().to(
											device) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else clip_label.type(
											torch.LongTensor).to(device)

										if "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
											representations, logits, c_sem = standard_inference(
												representation_architecture=representation_architecture,
												vit_model=vit_model,
												visual_features=visual_features,
												visual_mask=visual_mask, ids=ids,
												clip_label=None if no_cheating_test else clip_label,
												sampling_rate=0.)
										else:
											raise Exception(
												f"Unsupported representation_architecture for testing unlabeled set: {representation_architecture}")

										list_logits.append(logits)
										all_guids_np.append(guids)
										list_binary_targets.append(binary_targets)

								logits = torch.cat(list_logits)
								probabilities = torch.sigmoid(logits) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.softmax(logits, dim=-1)
								binary_targets = torch.cat(list_binary_targets)
								pred_label = (probabilities >= 0.5).int() if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.argmax(probabilities, dim=-1).int()

								probabilities_np = probabilities.detach().cpu().numpy().round(3)
								binary_targets_np = binary_targets.detach().cpu().numpy()
								pred_label_np = pred_label.detach().cpu().numpy().flatten()
								all_guids_np = np.concatenate(all_guids_np)
								pseudo_confidences_np = np.maximum(probabilities_np, 1-probabilities_np).flatten() if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else probabilities_np[np.arange(len(probabilities_np)), pred_label_np]

								# We reorder everything according to increasing pseudo-confidence
								pseudo_confidences_order = np.argsort(pseudo_confidences_np)
								binary_targets_np = binary_targets_np[pseudo_confidences_order]
								pred_label_np = pred_label_np[pseudo_confidences_order]
								all_guids_np = all_guids_np[pseudo_confidences_order]

								for f_b in also_relabel_val_set:
									print(f"fold_budget (for val set) : {f_b}")
									# We load the part of the df we want to relabel
									original_df = get_specific_df(gt_csv=gt_csv, classif_info=classif_info,
																  folds_info=folds_info,
																  fold_idx=fold_idx, annotator=train_annotator)
									original_df = original_df[original_df.imdb_key.isin(folds_info[fold_idx]["validation"])]
									# We proceed to actual relabeling based on the inference results above

									n_pseudo_labels_original_labels = binary_targets_np
									n_pseudo_labels = pred_label_np
									full_guids_np = all_guids_np

									new_pseudo_labels_mask = n_pseudo_labels != n_pseudo_labels_original_labels
									# If we exceed the remaining budget, we restrict the set of "new" pseudo-labels,
									# keeping only the most pseudo-confident
									if np.count_nonzero(new_pseudo_labels_mask) > f_b:
										true_idx = np.flatnonzero(new_pseudo_labels_mask)  # the True indices, ie. the indices that will cost some budget
										# We set to False all but the remaining_budget last changes (ie. all but the the most pseudo-confident)
										new_pseudo_labels_mask[true_idx[:(-f_b)]] = False
										# Note that the "unkept pseudo labels" observations are not thrown out,
										# we just keep the original labels instead
										# We update the n_pseudo_labels :
										n_pseudo_labels = np.where(new_pseudo_labels_mask, n_pseudo_labels,
																   n_pseudo_labels_original_labels)

									# Now we have the correct update, we recompute new_pseudo_label_mask for verbosity
									new_pseudo_labels_mask = n_pseudo_labels != n_pseudo_labels_original_labels
									print(f"Actual budget used : {np.count_nonzero(new_pseudo_labels_mask)}")
									# We update original_df

									# We replace the original labels by the pseudo-labels that were used at this training cycle
									# Note that, in EN_HNS exp for instance, all pseudo-labels HNS are set to 'Sure'
									classif_classes = classif_info["classif_classes"]
									if classif_info['type_classif'] == 'label':
										reverse_dico = {v: k for k, v in classif_classes.items()}
									else:
										raise Exception(f"classif_info type_classif {classif_info['type_classif']} not handled yet for pseudo-labeling")
									for guid, pseudo_label in zip(full_guids_np[new_pseudo_labels_mask], n_pseudo_labels[new_pseudo_labels_mask]):
										original_df.loc[original_df['guid'] == guid, 'label'] = reverse_dico[pseudo_label]

									# We save the relabeled originally unlabeled dataset
									csv_directory = f"{DATASETS_DIR}/relabeled_datasets/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{repr_str}_full_model/train_{train_annotator}/val_set_fold_budget_{f_b}"
									if not os.path.exists(csv_directory):
										os.makedirs(csv_directory)
									combined_csv_path = f"{csv_directory}/{dataset_name}_{train_annotator}_relabeled_val_seed_{seed}_fold_{fold_idx}.csv"
									original_df.to_csv(combined_csv_path, sep=';', index=False)
									print(f"\nSaved successfully the relabelled val set (for this fold and annotator) at {combined_csv_path}")

					else:
						training_cycles_finished = True

					del vit_model
					torch.cuda.empty_cache()

			# NOW WE HAVE BOTH REPRESENTATION MODELS AND (IF WE WANTED) !!! CORRESPONDING !!! CLASSIFICATION MODELS FOR ALL FOLDS

			# Here we do the actual cross-validation testing for cross-validation pseudo-labeling
			if pseudo_labeling and pseudo_labeling_val_to_unlabeled:
				print(f"\n\n\n### Cross-validation testing for pseudo-labeling")

				# Based on the saved "relabeled datasets", which are a partition of the original dataset,
				# we create the new-labeled dataset
				csv_directory = f"{DATASETS_DIR}/relabeled_datasets/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{repr_str}_full_model/train_{train_annotator}"
				fold_new_gt_csvs = [f"{csv_directory}/{dataset_name}_{train_annotator}_relabeled_seed_{seed}_fold_{fold_idx}.csv" for fold_idx in num_folds]
				fold_new_dfs = [pd.read_csv(f_csv, sep=";", converters={'concepts': pd.eval}) for f_csv in fold_new_gt_csvs]
				final_df = pd.concat(fold_new_dfs)
				new_gt_csv = f"{csv_directory}/{dataset_name}_{train_annotator}_relabeled_seed_{seed}.csv"
				final_df.to_csv(new_gt_csv, sep=';', index=False)
				print(f"\nSaved successfully the full relabelled set (annotator) at {new_gt_csv}")

				for fold_idx in num_folds:
					print(f"\n# FOLD_IDX {fold_idx}")
					device = set_gpu(gpu_index=gpu_index)
					# We retrieve the checkpoint path
					checkpoint_dir = f"{CHECKPOINTS_DIR}/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{train_annotator}"
					train_annotator_backbone = train_annotator if train_annotator_backbone is None else train_annotator_backbone
					last_global_model_path_representation = f"{checkpoint_dir}/best_ckpt_seed_{seed}_fold_{fold_idx}_repr_{repr_str}.pt" if pseudo_labeling else None
					# We load the model
					vit_model = load_representation_model(representation_architecture=representation_architecture,
														  global_model_path_representation=last_global_model_path_representation,
														  device=device,
														  # MODEL HYPERPARAMETERS
														  feature_dim=feature_dim, max_seq_len=max_seq_len,
														  dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
														  pool=pool, dim_head=dim_head,
														  dropout=dropout, emb_dropout=emb_dropout,
														  # To DISTINGUISH BETWEEN BINARY AND MULTICLASS CASE
														  num_output_neurons=num_output_neurons,
														  # SPECIFIC TO SOME MODELS ONLY
														  nb_domains=None,
														  backbone_path=None,
														  modality=modality)
					if vit_model is not None:  # for xclipemb
						vit_model.eval()

					# We load the test set
					test_dataset = FeatureDF(modality=modality, gt_csv=new_gt_csv, split="test", fold_idx=fold_idx,
											 folds_info=folds_info, classif_info=classif_info, max_seq_len=max_seq_len,
											 seed=seed, annotator=train_annotator, with_guid=True, with_id=True,
											 representation_architecture=representation_architecture,
											  specified_visual_feature_dir=specified_visual_feature_dir)
					test_loader = DataLoader(test_dataset, batch_size=len(
						test_dataset) if "differential" in representation_architecture else batch_size,
											 shuffle=False, num_workers=19,
											 collate_fn=my_collate_fn,)
					# We do the test inference on the test set. Note that, for now, only ViT and MLP models are supported
					list_logits = []
					list_binary_targets = []
					with ((torch.no_grad())):
						for i_val, tbatch in enumerate(test_loader):
							visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = tbatch
							clip_label, one_hot_concepts, ids = (
								clip_label.to(device),
								one_hot_concepts.to(device),
								ids.to(device))
							if multimodal:
								for i_mod in range(len(visual_features)):
									visual_features[i_mod] = visual_features[i_mod].to(device)
									visual_mask[i_mod] = visual_mask[i_mod].to(device)
							else:
								visual_features = visual_features.to(device)
								visual_mask = visual_mask.to(device)

							binary_targets = clip_label.float().to(device) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else clip_label.type(torch.LongTensor).to(device)

							if "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
								representations, logits, c_sem = standard_inference(
									representation_architecture=representation_architecture,
									vit_model=vit_model,
									visual_features=visual_features,
									visual_mask=visual_mask, ids=ids,
									clip_label=None if no_cheating_test else clip_label,
									sampling_rate=0.)
							else:
								raise Exception(
									f"Unsupported representation_architecture for testing unlabeled set: {representation_architecture}")

							list_logits.append(logits)
							list_binary_targets.append(binary_targets)

					logits = torch.cat(list_logits)
					probabilities = torch.sigmoid(logits) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.softmax(logits, dim=-1)
					probabilities_np = probabilities.detach().cpu().numpy().round(3)

					# We enjoy the occasion to print some verbosity
					binary_targets = torch.cat(list_binary_targets)
					binary_targets_np = binary_targets.detach().cpu().numpy()
					pred_label = (probabilities >= 0.5).int() if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.argmax(probabilities, dim=-1).int()
					pred_label_np = pred_label.detach().cpu().numpy()
					test_classif_metrics = calculate_tb_metrics(np.nan, binary_targets_np, pred_label_np,
																probabilities_np,
																num_output_neurons=num_output_neurons)
					print(f"Evaluation of classification of the core model on test set :\n{test_classif_metrics}")

					# For finer processing of results later, we also save the per-observation results, in a similar fashion to infer.py
					original_df = get_specific_df(gt_csv=new_gt_csv, classif_info=classif_info, folds_info=folds_info,
												  fold_idx=fold_idx, annotator=train_annotator)
					original_df = get_test_inferred_df(original_df, probabilities_np)
					csv_directory = f"{PROJECT_DIR}/results_two_stage/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{repr_str}_full_model/train_{train_annotator}"
					if not os.path.exists(csv_directory):
						os.makedirs(csv_directory)
					combined_csv_path = f"{csv_directory}/test_results_seed_{seed}_{train_annotator}_{exp_name}_fold_{fold_idx}.csv"
					# N.B. : We can leave the test prefix, since the removal of pseudo_labeling_str in the suffix prevents overwriting
					original_df.to_csv(combined_csv_path, sep=';', index=False)
					print(f"\nSaved successfully per-observation results at {combined_csv_path}")

			# The full relabeling of val set
			if pseudo_labeling and (also_relabel_val_set is not None):

				for f_b in also_relabel_val_set:

					print(f"\n\n\n### Cross-validation testing for pseudo-labeling, f_b = {f_b}")

					# Based on the saved "relabeled datasets", which are a partition of the original dataset,
					# we create the new-labeled dataset
					csv_directory = f"{DATASETS_DIR}/relabeled_datasets/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{repr_str}_full_model/train_{train_annotator}/val_set_fold_budget_{f_b}"
					fold_new_gt_csvs = [f"{csv_directory}/{dataset_name}_{train_annotator}_relabeled_val_seed_{seed}_fold_{fold_idx}.csv" for fold_idx in num_folds]
					fold_new_dfs = [pd.read_csv(f_csv, sep=";", converters={'concepts': pd.eval}) for f_csv in fold_new_gt_csvs]
					final_df = pd.concat(fold_new_dfs)
					new_gt_csv = f"{csv_directory}/{dataset_name}_{train_annotator}_relabeled_seed_{seed}.csv"
					final_df.to_csv(new_gt_csv, sep=';', index=False)
					print(f"\nSaved successfully the full relabelled set (annotator) at {new_gt_csv}")

					for fold_idx in num_folds:
						print(f"\n# FOLD_IDX {fold_idx}")
						device = set_gpu(gpu_index=gpu_index)
						# We retrieve the checkpoint path
						checkpoint_dir = f"{CHECKPOINTS_DIR}/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{train_annotator}"
						train_annotator_backbone = train_annotator if train_annotator_backbone is None else train_annotator_backbone
						last_global_model_path_representation = f"{checkpoint_dir}/best_ckpt_seed_{seed}_fold_{fold_idx}_repr_{repr_str}.pt" if pseudo_labeling else None
						# We load the model
						vit_model = load_representation_model(representation_architecture=representation_architecture,
															  global_model_path_representation=last_global_model_path_representation,
															  device=device,
															  # MODEL HYPERPARAMETERS
															  feature_dim=feature_dim, max_seq_len=max_seq_len,
															  dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
															  pool=pool, dim_head=dim_head,
															  dropout=dropout, emb_dropout=emb_dropout,
															  # To DISTINGUISH BETWEEN BINARY AND MULTICLASS CASE
															  num_output_neurons=num_output_neurons,
															  # SPECIFIC TO SOME MODELS ONLY
															  nb_domains=None,
															  backbone_path=None,
															  modality=modality)
						if vit_model is not None:  # for xclipemb
							vit_model.eval()

						# We load the test set
						test_dataset = FeatureDF(modality=modality, gt_csv=new_gt_csv, split="test", fold_idx=fold_idx,
												 folds_info=folds_info, classif_info=classif_info, max_seq_len=max_seq_len,
												 seed=seed, annotator=train_annotator, with_guid=True, with_id=True,
												 representation_architecture=representation_architecture,
											  specified_visual_feature_dir=specified_visual_feature_dir)
						test_loader = DataLoader(test_dataset, batch_size=len(
							test_dataset) if "differential" in representation_architecture else batch_size,
												 shuffle=False, num_workers=19,
												 collate_fn=my_collate_fn)
						# We do the test inference on the test set. Note that, for now, only ViT and MLP models are supported
						list_logits = []
						list_binary_targets = []
						with ((torch.no_grad())):
							for i_val, tbatch in enumerate(test_loader):
								visual_features, visual_mask, clip_label, one_hot_concepts, guids, ids = tbatch
								clip_label, one_hot_concepts, ids = (
									clip_label.to(device),
									one_hot_concepts.to(device),
									ids.to(device))
								if multimodal:
									for i_mod in range(len(visual_features)):
										visual_features[i_mod] = visual_features[i_mod].to(device)
										visual_mask[i_mod] = visual_mask[i_mod].to(device)
								else:
									visual_features = visual_features.to(device)
									visual_mask = visual_mask.to(device)

								binary_targets = clip_label.float().to(device) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else clip_label.type(torch.LongTensor).to(device)

								if "vit" in representation_architecture or "mlp" in representation_architecture or "cem" in representation_architecture:
									representations, logits, c_sem = standard_inference(
										representation_architecture=representation_architecture,
										vit_model=vit_model,
										visual_features=visual_features,
										visual_mask=visual_mask, ids=ids,
										clip_label=None if no_cheating_test else clip_label,
										sampling_rate=0.)
								else:
									raise Exception(
										f"Unsupported representation_architecture for testing unlabeled set: {representation_architecture}")

								list_logits.append(logits)
								list_binary_targets.append(binary_targets)

						logits = torch.cat(list_logits)
						probabilities = torch.sigmoid(logits) if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.softmax(logits, dim=-1)
						probabilities_np = probabilities.detach().cpu().numpy().round(3)

						# We enjoy the occasion to print some verbosity
						binary_targets = torch.cat(list_binary_targets)
						binary_targets_np = binary_targets.detach().cpu().numpy()
						pred_label = (probabilities >= 0.5).int() if num_output_neurons == 1 or classif_info['type_classif'] == 'several_concepts' else torch.argmax(probabilities, dim=-1).int()
						pred_label_np = pred_label.detach().cpu().numpy()
						test_classif_metrics = calculate_tb_metrics(np.nan, binary_targets_np, pred_label_np,
																	probabilities_np,
																	num_output_neurons=num_output_neurons)
						print(f"Evaluation of classification of the core model on test set :\n{test_classif_metrics}")

						# For finer processing of results later, we also save the per-observation results, in a similar fashion to infer.py
						original_df = get_specific_df(gt_csv=new_gt_csv, classif_info=classif_info, folds_info=folds_info,
													  fold_idx=fold_idx, annotator=train_annotator)
						original_df = get_test_inferred_df(original_df, probabilities_np)
						csv_directory = f"{PROJECT_DIR}/results_two_stage/{modality}/{exp_name}/{pool}{spec_str}_f_b_{f_b}/{dataset_name}/{repr_str}_full_model/train_{train_annotator}"
						if not os.path.exists(csv_directory):
							os.makedirs(csv_directory)
						combined_csv_path = f"{csv_directory}/test_results_seed_{seed}_{train_annotator}_{exp_name}_fold_{fold_idx}.csv"
						# N.B. : We can leave the test prefix, since the removal of pseudo_labeling_str in the suffix prevents overwriting
						original_df.to_csv(combined_csv_path, sep=';', index=False)
						print(f"\nSaved successfully per-observation results at {combined_csv_path}")

			# Here we save the target_target_train_losses (N.B. : this will only be used for computing training dynamics)
			if save_target_train_losses:
				ttl_directory = f'{PROJECT_DIR}/target_train_losses/modality/{modality}/{exp_name}/{pool}/{dataset_name}/{train_annotator}'
				# The directory contains all necessary information to characterize the model the target losses refer to
				# The only information that may be of importance is the balancing, so we specify it in the .json name
				json_name = f"{repr_str}_target_train_losses"
				if not os.path.exists(ttl_directory):
					os.makedirs(ttl_directory)
				with open(f'{ttl_directory}/{json_name}.json', 'w') as file:
					json.dump({'target_train_losses': annotator_target_train_losses}, file, indent=4)
				print(f'{ttl_directory}/{json_name}.json saved successfully')

			if evaluate_representation:
				# CV results of representation
				print(
					f"\n\nCV mean representation evaluation for model {repr_str} (on annotator {train_annotator} and seed {seed}) :")
				for type_prober in type_probers:
					cv_results = cv_results_all[type_prober]
					global_results = dict()
					global_results['prober'] = cv_results[0]['prober']
					# The fields we want to average
					all_names_dict = [name_dict for i in range(len(cv_results)) for name_dict in
									  list(cv_results[i].keys())]
					all_names_dict = list(set(all_names_dict).difference(["prober"]))
					considered_metrics = ['accuracy', f'precision', f'recall', f'pr_auc']
					for name_dict in all_names_dict:
						global_results[name_dict] = {
							k: round(float(np.nanmean([d[name_dict][k] for d in cv_results if name_dict in d.keys()])),
									 3) for k in considered_metrics}
					# Printing of those results
					print(f"\n{global_results}")
					# Saving of those results for easy future manipulation
					repr_metrics_dir = f'{PROJECT_DIR}/representation_metrics/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{repr_str}_{train_annotator}_{seed}'
					if not os.path.exists(repr_metrics_dir):
						os.makedirs(repr_metrics_dir)
					with open(f'{repr_metrics_dir}/{type_prober}.json', 'w') as file:
						json.dump(global_results, file, indent=4)

			if consider_classif:
				# CV results of classification
				print(
					f"\n\nCV mean classification evaluation for model {type_classif_str} (on top of {repr_str}) (on annotator {train_annotator} and seed {seed}) :")
				classif_results = {k: round(float(np.nanmean([d[k] for d in cv_results_classif])), 3) for k in
								   cv_results_classif[0].keys()}
				# Printing of those results
				print(f"\n{classif_results}")
				# Saving of those results for easy future manipulation
				repr_classif_metrics_dir = f'{PROJECT_DIR}/representation_classification_metrics/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{type_classif_str}_{repr_str}_{train_annotator}_{seed}'
				if not os.path.exists(repr_classif_metrics_dir):
					os.makedirs(repr_classif_metrics_dir)
				with open(f'{repr_classif_metrics_dir}/classif_metrics.json', 'w') as file:
					json.dump(classif_results, file, indent=4)

			if evaluate_representation_classifier and len(cv_results_classif_core) > 0:
				# CV results of classification on the full core model
				print(
					f"\n\nCV mean classification evaluation for the full model {repr_str} (on annotator {train_annotator} and seed {seed}) :")
				classif_results = {k: round(float(np.nanmean([d[k] for d in cv_results_classif_core])), 3) for k in
								   cv_results_classif_core[0].keys()}
				# Printing of those results
				print(f"\n{classif_results}")
				# Saving of those results for easy future manipulation
				repr_classif_metrics_dir = f'{PROJECT_DIR}/representation_classification_metrics/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{repr_str}_full_model_{train_annotator}_{seed}'
				if not os.path.exists(repr_classif_metrics_dir):
					os.makedirs(repr_classif_metrics_dir)
				with open(f'{repr_classif_metrics_dir}/classif_metrics.json', 'w') as file:
					json.dump(classif_results, file, indent=4)
