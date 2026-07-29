import os
import numpy as np
import json
import time
from torch.utils.data import DataLoader
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import homogeneity_score
from sklearn_extra.cluster import KMedoids
import scipy.stats as stats

from dataset.dataset import FeatureDF
from two_stage.two_stage_train import get_repr_strs
from global_utils import *
from constants import *


def visualize_representations(#CHARACTERIZING THE REPRESENTATION MODEL
		seed, train_annotator, fold_idx, pool, spec_str, exp_name,
		# TYPE OF REPRESENTATION TRAINING
		representation_architecture, representation_loss, batch_size,
		# CONTRASTIVE ARGUMENTS
		global_contrastive, contrastive_type, weighted_contrastive, temperature,
		# FOR DIFFERENTIAL (REPRESENTATION) TRAINING # for now only implemented in differential_ce
		nb_movies_per_batch=None, nb_clips_per_movie=None,
		# MIL
		nb_segment=None, length_segment=None,
		# OTHERS
		train_annotator_backbone=None, representation_backbone=None,
		# TO LOAD THE DATA TO REPRESENT
		modality=None, gt_csv=None, folds_info=None, classif_info=None,
		# TO SPECIFY THE DATA TO REPRESENT
		considered_annotator=None, considered_set=None,
		normalize_by_undersampling=False,
		# OTHERS
		gpu_index=None,
):
	# Adapt the feature_dir depending on the modality
	modality_to_dir = {"visual": VISUAL_FEATURE_DIR}
	feature_dir = modality_to_dir[modality]
	# Retrieve the dataset name (for precise naming of directories)
	dataset_name = gt_csv.split('/')[-1][:(-4)]

	normalize_str = '_balanced_movies' if normalize_by_undersampling else ''

	device = set_gpu(gpu_index=gpu_index)

	repr_str, type_representation = get_repr_strs(representation_architecture, representation_loss,
												  global_contrastive, contrastive_type, weighted_contrastive,
												  temperature,
												  nb_movies_per_batch, nb_clips_per_movie,
												  nb_segment, length_segment,
												  batch_size, train_annotator_backbone, representation_backbone,
												  untrained=False,
												  train_on_target=False, pseudo_labeling=False,
												  confidence_based_pseudo_labeling=False, budget=None
												  )

	# Checkpoint
	checkpoint_dir = f"{CHECKPOINTS_DIR}/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{train_annotator}"
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	global_model_path_representation = f"{checkpoint_dir}/best_ckpt_seed_{seed}_fold_{fold_idx}_repr_{repr_str}.pt"
	#existing_representation_model = os.path.exists(global_model_path_representation)
	backbone_repr_str, backbone_type_representation = get_repr_strs(**representation_backbone)
	backbone_path = f"{checkpoint_dir}/best_ckpt_seed_{seed}_fold_{fold_idx}_repr_{backbone_repr_str}.pt"
	# backbone_path will only be used by "aposteriori_differential_ce"

	# We define the model arguments (no need to make them arguments as they are always the same)
	max_seq_len = 160  # 480
	feature_dim = 512  # dimension of the features
	dim = 128  # 768 use bigger working dimension than feature dimension (according to the paper)
	heads = 8
	dim_head = dim // heads  # This ensures dim is divisible by heads
	mlp_dim = 1024  # 1536 #2024 Julie #1536  # 2 times the input dimension (2024 in the previous experiment)
	depth = 2  # 2 #6 original paper
	dropout = 0.1
	emb_dropout = 0.1

	nb_domains=None
	if "dann" in representation_architecture:
		if "dann_binary" in representation_architecture:
			nb_domains = 2
		elif "dann_by_prop" in representation_architecture:
			nb_domains = int(representation_architecture.split("_")[-1])
		else:
			nb_domains = 14

	# We load the model
	vit_model = load_representation_model(representation_architecture=representation_architecture,
										  global_model_path_representation=global_model_path_representation,
										  device=device,
										  # MODEL HYPERPARAMETERS
										  feature_dim=feature_dim, max_seq_len=max_seq_len,
										  dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
										  pool=pool, dim_head=dim_head,
										  dropout=dropout, emb_dropout=emb_dropout,
										  # SPECIFIC TO SOME MODELS ONLY
										  nb_domains=nb_domains,
										  backbone_path=backbone_path)

	# We load the data we want to represent (train, val and test)
	train_dataset = FeatureDF(feature_dir, gt_csv, split="train", fold_idx=fold_idx,
							folds_info=folds_info, classif_info=classif_info, max_seq_len=max_seq_len,
							seed=seed, annotator=considered_annotator, with_id=True,balancing=False,
							  representation_architecture=representation_architecture
							  )
	val_dataset = FeatureDF(feature_dir, gt_csv, split="validation", fold_idx=fold_idx,
							folds_info=folds_info, classif_info=classif_info, max_seq_len=max_seq_len,
							seed=seed, annotator=considered_annotator, with_id=True, balancing=False,
							  representation_architecture=representation_architecture
							)
	test_dataset = FeatureDF(feature_dir, gt_csv, split="test", fold_idx=fold_idx,
							 folds_info=folds_info, classif_info=classif_info, max_seq_len=max_seq_len,
							 seed=seed, annotator=considered_annotator, with_id=True, balancing=False,
							  representation_architecture=representation_architecture
							 )

	train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=19,
							collate_fn=custom_collate_fn_with_movie_ids)
	val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=19,
							collate_fn=custom_collate_fn_with_movie_ids)
	test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=19,
							 collate_fn=custom_collate_fn_with_movie_ids)
	# the batch size should simply be chosen so that it optimizes running time

	# From there we can build the representations
	start = time.time()

	if considered_set == "train":
		train_embeddings, train_labels, train_ids = (
			get_embeddings_labels_ids_from_loader(loader=train_loader, device=device,
												  representation_architecture=representation_architecture, vit_model=vit_model))
		embeddings, labels, ids = train_embeddings, train_labels, train_ids
	elif considered_set == "val":
		val_embeddings, val_labels, val_ids = (
			get_embeddings_labels_ids_from_loader(loader=val_loader, device=device,
												  representation_architecture=representation_architecture, vit_model=vit_model))
		embeddings, labels, ids = val_embeddings, val_labels, val_ids
	elif considered_set == "test":
		test_embeddings, test_labels, test_ids = (
			get_embeddings_labels_ids_from_loader(loader=test_loader, device=device,
												  representation_architecture=representation_architecture, vit_model=vit_model))
		embeddings, labels, ids = test_embeddings, test_labels, test_ids
	elif considered_set == "all":
		train_embeddings, train_labels, train_ids = (
			get_embeddings_labels_ids_from_loader(loader=train_loader, device=device,
												  representation_architecture=representation_architecture, vit_model=vit_model))
		val_embeddings, val_labels, val_ids = (
			get_embeddings_labels_ids_from_loader(loader=val_loader, device=device,
												  representation_architecture=representation_architecture, vit_model=vit_model))
		test_embeddings, test_labels, test_ids = (
			get_embeddings_labels_ids_from_loader(loader=test_loader, device=device,
												  representation_architecture=representation_architecture, vit_model=vit_model))
		embeddings, labels, ids = (torch.cat([train_embeddings, val_embeddings, test_embeddings]),
								   torch.cat([train_labels, val_labels, test_labels]),
								   torch.cat([train_ids, val_ids, test_ids]))
	else:
		raise Exception(f"Unknown considered_set : '{considered_set}'")

	# We convert those objects to numpy objects to easily use scikit-learn and pandas
	embeddings, int_labels, ids = embeddings.detach().cpu().numpy(), labels.detach().cpu().numpy(), ids.detach().cpu().numpy()
	# the renaming int_labels avoids accidental renaming

	end = time.time()
	print(f"\n\n## repr_{repr_str} : train_{train_annotator} (fold_{fold_idx}) on {considered_annotator}_{considered_set}")
	print(f"\ntime to compute representations : {round(end - start, 3)} s")
	print(f"train ({len(train_dataset)}) / val ({len(val_dataset)}) / test ({len(test_dataset)}) -> "
		  f"we represent {considered_set} ({len(int_labels)})")
	print(f"embeddings.shape : {embeddings.shape} ; int_labels.shape :{int_labels.shape} ; ids.shape : {ids.shape}")

	# We also load the corresponding .csv : in order to easily access all features of an observation, this is the simplest
	# Again, for this to make sense it is important that we do not shuffle the datasets
	original_df = get_specific_df(gt_csv, classif_info, folds_info, fold_idx, considered_annotator, restrict_to=considered_set)
	# for more explicit plots, we map the fold attribute to its actual meaning
	fold_num_to_name = {0: 'test', 1: 'val', 2: 'train'}
	original_df['fold'] = original_df['fold'].map(fold_num_to_name)

	# Now we can compute t-SNE
	# The standard t-SNE metric is euclidean, but in our case, for the plots to be consistent with
	# similarity computations, 'cosine' seems more appropriate.
	# Empirically, in our case, the choice of the metric has little impact
	# A possibility mentioned in scikit-learn documentation is to perform PCA beforehand for high-dimension data
	# (to avoid the curse of dimensionality, from my understanding). Our data has dimension 128.
	# Anyway cosine similarity solves the problem by itself.
	# "perplexity can be interpreted as a smooth measure of the effective number of neighbors.
	# The performance of SNE is fairly robust to changes in the perplexity, and typical values are between 5 and 50"
	# (So, the larger the data, the larger the perplexity should be.)
	# Empirically, we observe the mentioned robustness ; higher values make the clusters
	# further away from each other (which is good) but denser (so less detailed)
	# perplexity=10 seems like a good compromise (possibly 20 to distinguish the clusters even better)
	metric='cosine'
	perplexity=10

	start = time.time()
	tsne_embeddings = TSNE(n_components=2,
						   metric=metric,
						   perplexity=perplexity,
						   learning_rate='auto',
						   init='pca', # more stable according to documentation ; this should have little impact anyway
						   ).fit_transform(embeddings)
	end = time.time()
	#print(f"\ntime to compute t-SNE embeddings : {round(end - start, 3)} s")
	#print(f"tsne_embeddings.shape : {tsne_embeddings.shape}")

	# Before doing anything else, if we normalize by undersampling, we perform simultaneous undersampling on
	# original_df, embeddings, tsne_embeddings, int_labels
	if normalize_by_undersampling:
		print(f"\n normalize_by_undersampling !!\n")
		list_movie_df = []
		list_embeddings = []
		list_tsne_embeddings = []
		list_int_labels = []
		for movie_title in original_df.movie_title.unique():
			movie_df = original_df[original_df.movie_title == movie_title]
			movie_embeddings = embeddings[original_df.movie_title == movie_title]
			movie_tsne_embeddings = tsne_embeddings[original_df.movie_title == movie_title]
			movie_int_labels = int_labels[original_df.movie_title == movie_title]
			diff_labels, int_label_counts = np.unique(movie_int_labels, return_counts=True)
			min_count = np.min(int_label_counts)

			int_label_list_movie_df = []
			int_label_list_embeddings = []
			int_label_list_tsne_embeddings = []
			int_label_list_int_labels = []

			for int_label in diff_labels:
				int_label_movie_df = movie_df[movie_int_labels == int_label]
				int_label_movie_embeddings = movie_embeddings[movie_int_labels == int_label]
				int_label_movie_tsne_embeddings = movie_tsne_embeddings[movie_int_labels == int_label]
				int_label_movie_int_labels = movie_int_labels[movie_int_labels == int_label]
				selected_indices = np.random.choice(np.arange(len(int_label_movie_int_labels)), size=min_count, replace=False)

				int_label_movie_df = int_label_movie_df.iloc[selected_indices]
				int_label_movie_embeddings = int_label_movie_embeddings[selected_indices]
				int_label_movie_tsne_embeddings = int_label_movie_tsne_embeddings[selected_indices]
				int_label_movie_int_labels = int_label_movie_int_labels[selected_indices]

				int_label_list_movie_df.append(int_label_movie_df)
				int_label_list_embeddings.append(int_label_movie_embeddings)
				int_label_list_tsne_embeddings.append(int_label_movie_tsne_embeddings)
				int_label_list_int_labels.append(int_label_movie_int_labels)

			movie_df = pd.concat(int_label_list_movie_df)
			movie_embeddings = np.concatenate(int_label_list_embeddings)
			movie_tsne_embeddings = np.concatenate(int_label_list_tsne_embeddings)
			movie_int_labels = np.concatenate(int_label_list_int_labels)

			list_movie_df.append(movie_df)
			list_embeddings.append(movie_embeddings)
			list_tsne_embeddings.append(movie_tsne_embeddings)
			list_int_labels.append(movie_int_labels)

		original_df = pd.concat(list_movie_df)
		embeddings = np.concatenate(list_embeddings)
		tsne_embeddings = np.concatenate(list_tsne_embeddings)
		int_labels = np.concatenate(list_int_labels)

	#print(f"\nAfter normalization by undersampling :")
	#print(f"len(original_df) : {len(original_df)}")
	#print(f"embeddings.shape : {embeddings.shape}")
	#print(f"tsne_embeddings.shape : {tsne_embeddings.shape}")
	#print(f"int_labels.shape : {int_labels.shape}")

	original_df['t-SNE x'] = tsne_embeddings[:, 0]
	original_df['t-SNE y'] = tsne_embeddings[:, 1]

	#Now original_df contains all necessary information, so we can perform plotting
	# We do hue/style according to various characteristics
	movies = original_df.movie_title.value_counts().sort_index(ascending=True).index.to_list()
	folds = ['train', 'val', 'test']
	labels = original_df.label.value_counts().sort_index(ascending=True).index.to_list()
	annotators = original_df.annotator.value_counts().sort_index(ascending=True).index.to_list()
	name_to_order = {'fold': folds, 'movie_title': movies, 'label': labels, 'annotator': annotators}

	plots_to_do = [('fold', 'movie_title'), ('movie_title', 'label'), ('label', 'fold'), ('fold', 'annotator')]
	if len(annotators) > 1:
		plots_to_do += [('annotator', 'movie_title')]

	for style, hue in plots_to_do:
		style_order = name_to_order[style] if style in name_to_order.keys() else None
		hue_order = name_to_order[hue] if hue in name_to_order.keys() else None
		ax = sns.relplot(data=original_df, x='t-SNE x', y='t-SNE y',
						 style=style,
						 hue=hue,
						 style_order=style_order,
						 hue_order=hue_order,
						 s=10)
		plt.setp(ax._legend.get_texts(), fontsize='5') # so that the legend fits in the plot
		plot_dir = (f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{train_annotator}/tsne_representation/"
					f"on_{style}_{hue}")
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)
		plt.savefig(f"{plot_dir}/fold_{fold_idx}_repr_{repr_str}_on_{considered_annotator}_{considered_set}{normalize_str}.png",
					dpi=200)
		plt.close()

	# Also, the "style" is not really helpful and can even harm the readability, so let's do some plots only on hue
	for hue in name_to_order.keys():
		hue_order = name_to_order[hue] if hue in name_to_order.keys() else None
		ax = sns.relplot(data=original_df, x='t-SNE x', y='t-SNE y',
						 hue=hue,
						 hue_order=hue_order,
						 s=10)
		plot_dir = (f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{train_annotator}/tsne_representation/"
					f"on_{hue}")
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)
		plt.savefig(f"{plot_dir}/fold_{fold_idx}_repr_{repr_str}_on_{considered_annotator}_{considered_set}{normalize_str}.png",
					dpi=200)
		plt.close()

	# We also represent the hue according to concepts:
	for concept in ALL_CONCEPTS:
		original_df[concept] = original_df['concepts'].apply(lambda clip_concepts: any(x in [concept] for x in clip_concepts)).astype(int)
		hue_order = [0,1]
		ax = sns.relplot(data=original_df, x='t-SNE x', y='t-SNE y',
						 hue=concept,
						 hue_order=hue_order,
						 s=10)
		plot_dir = (f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{train_annotator}/tsne_representation/"
					f"on_{concept}")
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)
		plt.savefig(f"{plot_dir}/fold_{fold_idx}_repr_{repr_str}_on_{considered_annotator}_{considered_set}{normalize_str}.png",
					dpi=200)
		plt.close()

	# If the considered set is the test set, we can also plot the predictions
	if considered_set == "test" and not normalize_by_undersampling:
		model_name = f'{repr_str}_full_model' if ('ce' in repr_str or 'dann' in repr_str) else None # for now we only treat full models
		test_annotator = considered_annotator
		csv_directory = f"{PROJECT_DIR}/results_two_stage/{modality}/{exp_name}/{pool}{spec_str}/{dataset_name}/{model_name}/train_{train_annotator}"
		combined_csv_path = f"{csv_directory}/test_results_seed_{seed}_{test_annotator}_{exp_name}_fold_{fold_idx}.csv"

		# We load the result df for this fold
		result_df = pd.read_csv(combined_csv_path, sep=";", converters={'concepts': pd.eval})
		# Convert probabilities to binary predictions using the threshold
		result_df['predictions'] = (result_df['test_proba'] >= 0.5).astype(int)
		print(f"len(result_df) : {len(result_df)}")
		result_df['t-SNE x'] = tsne_embeddings[:, 0]
		result_df['t-SNE y'] = tsne_embeddings[:, 1]
		# Now we can create the column joining train and test label (true and predicted label)
		def get_joint_label(row):
			return f"{row.label_int}_true_{row.predictions}_pred"
		result_df["true_pred_labels"] = result_df.apply(get_joint_label, axis=1, result_type='reduce')
		# Now we proceed to plotting
		ax = sns.relplot(data=result_df, x='t-SNE x', y='t-SNE y',
						 hue="true_pred_labels",
						 hue_order=sorted(list(result_df["true_pred_labels"].unique())),
						 s=10)
		plot_dir = (f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{train_annotator}/tsne_representation/"
					f"on_true_pred_labels")
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)
		plt.savefig(
			f"{plot_dir}/fold_{fold_idx}_repr_{repr_str}_on_{considered_annotator}_{considered_set}{normalize_str}.png",
			dpi=200)
		plt.close()

		# Also representing annotator
		ax = sns.relplot(data=result_df, x='t-SNE x', y='t-SNE y',
						 hue="true_pred_labels",
						 hue_order=sorted(list(result_df["true_pred_labels"].unique())),
						 style='annotator',
						 style_order=annotators,
						 s=10)
		plot_dir = (f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{train_annotator}/tsne_representation/"
					f"on_true_pred_labels_annotator")
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)
		plt.savefig(
			f"{plot_dir}/fold_{fold_idx}_repr_{repr_str}_on_{considered_annotator}_{considered_set}{normalize_str}.png",
			dpi=200)
		plt.close()

	# Based on the embeddings (or tsne_embeddings), we compute the clustering metrics of interest
	# To avoid quasi-duplicates that could noise the results, better to test it on train_annotator = test_annotator

	# First, we create the "labels" that will be associated to movies
	movie_labels = dict()
	for movie_title in original_df.movie_title.unique():
		movie_labels[movie_title] = (original_df.movie_title == movie_title).to_numpy().astype(int)
	# We also create the "multiclass" movie labels (because a priori it may also be relevant)
	title_to_id = {str(movie_title): i for i, movie_title in enumerate(original_df.movie_title.unique())}
	multiclass_movie_labels = original_df.movie_title.map(title_to_id).to_numpy().astype(int)
	# We also create the concept labels
	concept_labels = dict()
	for concept in ALL_CONCEPTS:
		concept_labels[concept] = original_df[concept].to_numpy().astype(int)

	print(f"\nThere are {len(embeddings)} observations : take this into account to choose 'num_n_clusters'") # ca. 500 for test
	num_n_clusters = 50
	# From my understanding, averaging over the possible values of n_clusters
	# spares from deciding a value of n_cluster beforehand... especially as clustering may not be obvious for any value
	n_clusters = np.linspace(2, len(embeddings), num=num_n_clusters).astype(int)

	# We will keep track of the various homogeneities
	movie_multiclass_homogeneity = []
	movie_multiclass_homogeneity_normalized = []
	movie_binary_homogeneity = []
	movie_binary_homogeneity_normalized = []
	label_homogeneity = []
	movies_binary_homogeneity = {str(movie_title): [] for movie_title in original_df.movie_title.unique()}
	movies_binary_homogeneity_normalized = {str(movie_title): [] for movie_title in original_df.movie_title.unique()}
	concepts_homogeneity = {concept: [] for concept in ALL_CONCEPTS}

	# Inspired from https://github.com/mateoespinosa/cem/blob/main/cem/metrics/cas.py
	# The core difference with CAS code is that here, the embedding is "shared" for all movies,
	# unlike concepts from CAS, there is no need to recompute a clustering for each of them
	for nc in n_clusters:
		kmedoids = KMedoids(n_clusters=nc, metric='cosine', random_state=0)
		# if too slow, can try metric == 'precomputed', and give the distance matrix during the fit call
		# I may try other methods later, maybe hierarchical is faster than looping over n_cluster
		start = time.time()
		cluster_labels = kmedoids.fit_predict(embeddings)
		end = time.time()
		#print(f"Training K-medoids for {nc} clusters : {round(end-start, 3)} s")
		movie_label_homogeneity = homogeneity_score(int_labels, multiclass_movie_labels)
		label_homogeneity.append(homogeneity_score(int_labels, cluster_labels))
		movie_multiclass_homogeneity.append(homogeneity_score(multiclass_movie_labels, cluster_labels))
		movie_multiclass_homogeneity_normalized.append(homogeneity_score(multiclass_movie_labels, cluster_labels) - movie_label_homogeneity)
		for movie_title in original_df.movie_title.unique():
			movie_homogeneity = homogeneity_score(movie_labels[movie_title], cluster_labels)
			movie_binary_homogeneity.append(movie_homogeneity)
			# N.B. : there is no need to compute the "sub-mean", because nb_movies is the same for each n_cluster,
			# so the mean of means is still the mean
			movies_binary_homogeneity[movie_title].append(movie_homogeneity)
			# We also handle the normalization relatively to the proportion of positive and negative
			movie_int_labels = int_labels[movie_labels[movie_title].astype(bool)]
			movie_pos_prop = np.count_nonzero(movie_int_labels) / len(movie_int_labels) # N.B. :designed for binary labels
			movie_majority_class_prop = max(movie_pos_prop, 1-movie_pos_prop)
			normalizing_factor = 1 - (movie_majority_class_prop - 0.5) / 0.5 # equal to 1 if labels are balanced in the movie
			movie_binary_homogeneity_normalized.append(normalizing_factor * movie_homogeneity)
			movies_binary_homogeneity_normalized[movie_title].append(normalizing_factor * movie_homogeneity)

		for concept in ALL_CONCEPTS:
			concepts_homogeneity[concept].append(homogeneity_score(concept_labels[concept], cluster_labels))

	# We can now compute the average over the possible n_cluster's :
	movie_multiclass_homogeneity = float(np.mean(movie_multiclass_homogeneity))
	movie_binary_homogeneity = float(np.mean(movie_binary_homogeneity))
	movie_multiclass_homogeneity_normalized = float(np.mean(movie_multiclass_homogeneity_normalized))
	movie_binary_homogeneity_normalized = float(np.mean(movie_binary_homogeneity_normalized))
	label_homogeneity = float(np.mean(label_homogeneity))
	for movie_title in original_df.movie_title.unique():
		movies_binary_homogeneity[movie_title] = float(np.mean(movies_binary_homogeneity[movie_title]))
		movies_binary_homogeneity_normalized[movie_title] = float(np.mean(movies_binary_homogeneity_normalized[movie_title]))
	for concept in ALL_CONCEPTS:
		concepts_homogeneity[concept] = float(np.mean(concepts_homogeneity[concept]))

	# We can save the corresponding dict :
	homogeneity_dict = {'movie_multiclass_homogeneity': movie_multiclass_homogeneity,
						'movie_binary_homogeneity': movie_binary_homogeneity,
						'movie_multiclass_homogeneity_normalized': movie_multiclass_homogeneity_normalized,
						'movie_binary_homogeneity_normalized': movie_binary_homogeneity_normalized,
						'label_homogeneity': label_homogeneity,
						'movies_binary_homogeneity': movies_binary_homogeneity,
						'movies_binary_homogeneity_normalized': movies_binary_homogeneity_normalized,
						'concepts_homogeneity': concepts_homogeneity}
	saving_dir = f"{PROJECT_DIR}/representation_homogeneities/{exp_name}/train_{train_annotator}_seed_{seed}"
	if not os.path.exists(saving_dir):
		os.makedirs(saving_dir)
	with open(f'{saving_dir}/fold_{fold_idx}_repr_{repr_str}_on_{considered_annotator}_{considered_set}{normalize_str}.json', 'w') as file:
		json.dump(homogeneity_dict, file, indent=4)

	# We also enjoy the occasion to do a qualitative analysis, and highlight interesting observations of the t-SNE map
	original_df = original_df.drop(columns=['subtitle', 'comment', 'framerate'] + ALL_CONCEPTS)
	nb_extremes = 5
	for axis in ['x', 'y']:
		sorted_indices = np.argsort(original_df[f't-SNE {axis}'])
		for extreme in ['min', 'max']:
			extreme_ids = sorted_indices[:nb_extremes] if extreme == 'min' else sorted_indices[(-nb_extremes):]
			print(f'\n{axis} {extreme} :\n{original_df.iloc[extreme_ids].to_string()}')

	# We treat the specific case of The Help
	nb_neighbors = 5
	if 'The Help' in movies:
		print(f"\nThe Help clusters (characterized by cluster centers) :")
		movie_title = 'The Help'
		the_help_df = original_df[original_df.movie_title == movie_title]
		the_help_embeddings = embeddings[original_df.movie_title == movie_title]
		the_help_tsne_embeddings = tsne_embeddings[original_df.movie_title == movie_title]
		the_help_int_labels = int_labels[original_df.movie_title == movie_title]
		normalized_the_help_tsne_embeddings = the_help_tsne_embeddings / np.linalg.norm(the_help_tsne_embeddings, axis=1, keepdims=True)
		the_help_similarities = normalized_the_help_tsne_embeddings @ normalized_the_help_tsne_embeddings.T
		#print(f"the_help_similarities : {the_help_similarities.shape}")

		kmedoids = KMedoids(n_clusters=3, metric='cosine', random_state=0)
		cluster_labels = kmedoids.fit_predict(the_help_tsne_embeddings)
		cluster_centers = kmedoids.medoid_indices_
		#print(f"cluster_centers : {cluster_centers}")
		for cluster, cluster_center in enumerate(cluster_centers):
			neighbor_indices = np.argsort(the_help_similarities[cluster_center])[(- nb_neighbors):]
			print(f'\ncluster {cluster} : neighbor_indices : {neighbor_indices}\n{the_help_df.iloc[neighbor_indices].to_string()}')

		# To be sure the clustering on The Help makes sense, we plot it
		the_help_df["cluster_labels"] = cluster_labels
		ax = sns.relplot(data=the_help_df, x='t-SNE x', y='t-SNE y',
						 hue="cluster_labels",
						 hue_order=[0,1,2],
						 s=10)
		plt.setp(ax._legend.get_texts(), fontsize='5')  # so that the legend fits in the plot
		plot_dir = (f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{train_annotator}/tsne_representation/"
					f"on_The_Help_clustering")
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)
		plt.savefig(
			f"{plot_dir}/fold_{fold_idx}_repr_{repr_str}_on_{considered_annotator}_{considered_set}{normalize_str}.png",
			dpi=200)
		plt.close()


if __name__ == "__main__":
	seed=42
	exp_name='EN_HNS'
	seeds = [seed]

	considered_set = 'test'
	normalize_by_undersampling=False

	nb_segment = 16
	length_segment = 6
	representation_loss = 'CE'

	for seed in seeds:
		seed = int(seed)
		for representation_architecture in ['vit', 'differential_vit']:
			for train_annotator in ["annotator_2"]:
				for considered_annotator in ["annotator_2"]: # ["annotator_2", "annotator_1", "UNION"]
					for fold_idx in np.arange(5):
						visualize_representations(  # CHARACTERIZING THE REPRESENTATION MODEL
							seed, train_annotator=train_annotator, fold_idx=fold_idx, pool="cls-pooling", spec_str="_two_stage", exp_name=exp_name,
							# TYPE OF REPRESENTATION TRAINING
							representation_architecture=representation_architecture, representation_loss=representation_loss, batch_size=128,
							# CONTRASTIVE ARGUMENTS
							global_contrastive=None, contrastive_type=None, weighted_contrastive=None, temperature=None,
							# FOR DIFFERENTIAL (REPRESENTATION) TRAINING # for now only implemented in differential_vit
							nb_movies_per_batch=2, nb_clips_per_movie=32,
							# MIL
							nb_segment=nb_segment, length_segment=length_segment,
							# OTHERS
							train_annotator_backbone=None, representation_backbone=None,
							# TO LOAD THE DATA TO REPRESENT
							modality="visual",
							gt_csv=f"{DATASETS_DIR}/mobygaze_dataframe_visual.csv",
							folds_info=FOLDS_INFO, classif_info=classif_info_from_exp_name(exp_name),
							# TO SPECIFY THE DATA TO REPRESENT
							considered_annotator=considered_annotator, considered_set=considered_set,
							normalize_by_undersampling=normalize_by_undersampling,
							# OTHERS
							gpu_index=2,
						)
