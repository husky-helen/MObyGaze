from two_stage.analysis.analyze_two_stage_disaggregated_results import plot_task_barplot_accuracy
from two_stage.two_stage_train import get_repr_strs

for addon in ['', 'differential_', 'gated_differential_']:
	for inference_set in ['val', 'test']:
		exp_name = "EN_HNS"
		GLOBAL_TRAIN_ANNOTATOR = 'annotator_2'
		modality = 'visual'
		dataset_name = f'mobygaze_dataframe_soft_{modality}'
		seed = 42
		train_annotator = GLOBAL_TRAIN_ANNOTATOR
		test_annotator = GLOBAL_TRAIN_ANNOTATOR

		representation_loss = 'CE'
		core_architecture = 'vit'
		aposteriori_backbone = {'representation_architecture': core_architecture, 'representation_loss': "CE",
				'global_contrastive': None, 'contrastive_type': None, 'weighted_contrastive': None,
				'temperature': None,
				'nb_movies_per_batch': None, 'nb_clips_per_movie': None,
				'nb_segment': None, 'length_segment': None,
				'target_model': None,
				'batch_size': 128, 'train_annotator_backbone': None, 'representation_backbone': None}
		representation_architecture = f"{addon}{core_architecture}"
		nb_movies_per_batch = 2
		nb_clips_per_movie = 32
		batch_size = 128
		exp_dict = {'exp_name': exp_name, 'train_annotator': train_annotator,
					'seed': seed,
					# TYPE OF REPRESENTATION TRAINING
					'representation_architecture': representation_architecture,
					'representation_loss': representation_loss,
					'batch_size': batch_size,
					'full_model': True,
					# CONTRASTIVE ARGUMENTS
					'global_contrastive': None, 'contrastive_type': None,
					'weighted_contrastive': None, 'temperature': None,
					# TYPE OF CLASSIFICATION TRAINING
					'type_classif': None, 'batch_size_class': None,
					# DIFFERENTIAL TRAINING
					'nb_movies_per_batch': nb_movies_per_batch,
					'nb_clips_per_movie': nb_clips_per_movie,
					# MIL
					'nb_segment': None, 'length_segment': None,
					# TARGET EMBEDDING
					'target_model': None,
					# BACKBONE
					'train_annotator_backbone': None,
					'representation_backbone': aposteriori_backbone,
					# FOR DETAILED DIRECTORY
					'modality': modality, 'pool': 'cls-pooling', 'spec_str': '_DMLR',
					'dataset_name': dataset_name,
					}
		repr_str, type_representation = get_repr_strs(**exp_dict)
		exp_dict["type_representation"] = type_representation

		plot_task_barplot_accuracy(exp_dict, model_name=representation_architecture, task=exp_name,
								   test_annotator=test_annotator, inference_set=inference_set,
								   rotation=90)
