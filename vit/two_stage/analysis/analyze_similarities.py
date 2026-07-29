from torch.utils.data import DataLoader
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from dataset.dataset import FeatureDF
from two_stage.two_stage_train import get_repr_strs
from global_utils import *
from constants import *


def analyze_similarities(#CHARACTERIZING THE REPRESENTATION MODEL
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
		# TO LOAD THE DATA
		modality=None, gt_csv=None, folds_info=None, classif_info=None,
		# TO SPECIFY THE DATA TO REPRESENT
		cache_annotator=None, input_annotator=None,
		# the sets are determined "by definition" : train for the cache and test for the input
		# OTHERS
		print_mean_similarities=False,
		gpu_index=None,
):
	# Adapt the feature_dir depending on the modality
	modality_to_dir = {"visual": VISUAL_FEATURE_DIR}
	feature_dir = modality_to_dir[modality]
	# Retrieve the dataset name (for precise naming of directories)
	dataset_name = gt_csv.split('/')[-1][:(-4)]

	if train_annotator != cache_annotator:
		raise Exception(f"train_annotator '{train_annotator}' different from cache_annotator '{cache_annotator}' :"
						f"if this is really waht you want, please change code")

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

	nb_domains=None # since we do not train DANN here, we do not care about this value

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

	# We load the train data we want to use as cache
	train_dataset = FeatureDF(feature_dir, gt_csv, split="train", fold_idx=fold_idx,
							folds_info=folds_info, classif_info=classif_info, max_seq_len=max_seq_len,
							seed=seed, annotator=cache_annotator, with_id=True,balancing=False,
							  representation_architecture=representation_architecture)
	train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=19,
							  collate_fn=custom_collate_fn_with_movie_ids)

	# We load the test data we want to use as input
	test_dataset = FeatureDF(feature_dir, gt_csv, split="test", fold_idx=fold_idx,
							 folds_info=folds_info, classif_info=classif_info, max_seq_len=max_seq_len,
							 seed=seed, annotator=input_annotator, with_id=True, balancing=False,
							  representation_architecture=representation_architecture)
	test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=19,
							 collate_fn=custom_collate_fn_with_movie_ids)
	# the batch size should simply be chosen so that it optimizes running time

	# From there we can build the representations
	train_embeddings, train_labels, train_ids = (
		get_embeddings_labels_ids_from_loader(loader=train_loader, device=device,
											  representation_architecture=representation_architecture, vit_model=vit_model))
	test_embeddings, test_labels, test_ids = (
		get_embeddings_labels_ids_from_loader(loader=test_loader, device=device,
											  representation_architecture=representation_architecture, vit_model=vit_model))

	with torch.no_grad():
		train_cache = F.normalize(train_embeddings, dim=-1, p=2).to(device)
		test_input = F.normalize(test_embeddings, dim=-1, p=2).to(device)
		similarity = (test_input @ train_cache.T).detach().cpu().numpy() # size (nb_test, nb_train)

	# For selection of the appropriate rows/columns, we load the corresponding dfs :
	train_df = get_specific_df(gt_csv, classif_info, folds_info, fold_idx, cache_annotator,
								  restrict_to="train").drop(columns=['subtitle', 'comment', 'fold'])
	test_df = get_specific_df(gt_csv, classif_info, folds_info, fold_idx, input_annotator,
							   restrict_to="test").drop(columns=['subtitle', 'comment', 'fold'])

	# We construct the cross-df that will contain similarities : it will enable an easy selection with sns
	cross_df = test_df.merge(train_df, how="cross", suffixes=["_test", "_train"])
	# We construct the column encoding the joint train/test labels
	label_to_int = classif_info_from_exp_name(exp_name)['classif_classes']
	def get_joint_label(row):
		return f"{label_to_int[row.label_train]}_train_{label_to_int[row.label_test]}_test"
	cross_df["train_test_labels"] = cross_df.apply(get_joint_label, axis=1, result_type='reduce')
	# We initialize the similarity column to easily manipulate it
	cross_df['similarity'] = -10.  # init with -10.
	# Now we should fill the cross df with the corresponding similarity
	for index_train in range(len(train_df)):
		guid_train = train_df["guid"].iat[index_train]
		cross_df.loc[cross_df['guid_train'] == guid_train, "similarity"] = similarity[:,index_train]
	# N.B. : we exploit the fact that the order of left keys is preserved

	# From there, we can compute/plot the distributions we want based on similarity
	test_movies = test_df.movie_title.value_counts().sort_index(ascending=True).index.to_list()
	labels = test_df.label.value_counts().sort_index(ascending=True).index.to_list()
	annotators = test_df.annotator.value_counts().sort_index(ascending=True).index.to_list()
	train_test_labels = cross_df.train_test_labels.value_counts().sort_index(ascending=True).index.to_list()
	name_to_order = {'movie_title_test': test_movies, #'label_test': labels, 'annotator_test': annotators,
					 'train_test_labels': train_test_labels}

	# We do the global plot of all similarities
	sns.histplot(x='similarity', data=cross_df)
	# Saving
	plot_dir = (f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{cache_annotator}/test_{input_annotator}/"
				f"similarity_distribution")
	if not os.path.exists(plot_dir):
		os.makedirs(plot_dir)
	plt.savefig(f"{plot_dir}/fold_{fold_idx}_repr_{repr_str}_similarity_hist.png",
				dpi=200)
	plt.close()

	for hue in name_to_order.keys():
		hue_order = name_to_order[hue] if hue in name_to_order.keys() else None
		sns.histplot(x='similarity', data=cross_df, hue=hue, hue_order=hue_order, multiple="layer")
		# Saving
		plot_dir = (f"{PROJECT_DIR}/plots/{exp_name}/two_stage/train_{cache_annotator}/test_{input_annotator}/"
					f"similarity_distribution")
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)
		plt.savefig(f"{plot_dir}/fold_{fold_idx}_repr_{repr_str}_similarity_hist_{hue}.png",
					dpi=200)
		plt.close()

	# We also plot the results for further analysis :
	if print_mean_similarities:
		for train_test_label in sorted(list(cross_df["train_test_labels"].unique())):
			similarities = cross_df[cross_df["train_test_labels"] == train_test_label]["similarity"]
			print(f"{train_test_label} :\n{similarities.describe(percentiles=[])}")

	# We also print the discrepancy between the good and bad train distributions, for each test label
	# N.B. : the following code is for now designed for binary only

	# First we need to create columns for train and test label int
	def get_int_label_train(row):
		return label_to_int[row.label_train]
	cross_df["int_label_train"] = cross_df.apply(get_int_label_train, axis=1, result_type='reduce')
	def get_int_label_test(row):
		return label_to_int[row.label_test]
	cross_df["int_label_test"] = cross_df.apply(get_int_label_test, axis=1, result_type='reduce')

	# Moreover, the distributions are too big for an efficient implementation ;
	# for more efficient implementation, we restrict the length (it shouldn't influence the end result too much)
	max_obs = 10000

	label_ints = sorted(list(cross_df.int_label_test.unique()))
	for label_test in label_ints:
		print(f"\nlabel_test : {label_test}")
		bad_labels = list(set(label_ints).difference({label_test}))
		d_o = cross_df.query("int_label_test == @label_test and int_label_train == @label_test")[
			'similarity'].to_numpy().astype(np.half)
		d_o = np.random.choice(d_o, size=max_obs, replace=False)
		# d_o is the "good distribution" ; we name it so by analogy with the article presenting the sigma measure
		for bad_label in bad_labels:
			d_e = cross_df.query("int_label_test == @label_test and int_label_train == @bad_label")[
				'similarity'].to_numpy().astype(np.half)
			d_e = np.random.choice(d_e, size=max_obs, replace=False)
			# Unlike the sigma article, we expect the bad distribution to be on the left (similarities should be smaller),
			# so for a given similarity from the good distribution, "more extreme" means larger
			# list_pvalues = np.array([stats.ks_2samp([x], d_e, alternative="less").pvalue for x in d_o])
			# the line above is too costly (especially given the size of our distributions) :
			# instead we compute the frequentist implementation below
			list_pvalues = (np.subtract(d_e, d_o.reshape(-1, 1), dtype=np.half) >= 0).mean(axis=1)
			p_value_threshold = 0.05
			sigma = (list_pvalues <= p_value_threshold).sum() / len(d_o)
			ks_stat = stats.ks_2samp(d_o, d_e, alternative="less").statistic
			print(f"incorrect train label {bad_label} distribution distance to correct train label distribution :"
				  f"sigma : {sigma} ; K-S : {ks_stat}")


if __name__ == "__main__":
	seed=42
	exp_name='EN_HNS'

	nb_segment = 16
	length_segment = 6
	representation_loss = 'CE'

	print_mean_similarities = False

	for representation_architecture in ['vit', 'differential_vit']:
		print(f"\n## {representation_architecture}")
		for train_annotator in ["annotator_2", "annotator_1"]:
			for cache_annotator in [train_annotator]:
				for input_annotator in [train_annotator]: # ["annotator_2", "annotator_1"]
					print(f"\ntrain, test : {train_annotator}, {input_annotator}")
					for fold_idx in np.arange(5):
						print(f"\nfold {fold_idx}")
						analyze_similarities(  # CHARACTERIZING THE REPRESENTATION MODEL
							seed, train_annotator=train_annotator, fold_idx=fold_idx, pool="cls-pooling", spec_str="_two_stage", exp_name=exp_name,
							# TYPE OF REPRESENTATION TRAINING
							representation_architecture=representation_architecture,
							representation_loss=representation_loss, batch_size=128,
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
							cache_annotator=cache_annotator, input_annotator=input_annotator,
							# the sets are determined "by definition" : train for the cache and test for the input
							# OTHERS
							print_mean_similarities=print_mean_similarities,
							gpu_index=0,
						)
