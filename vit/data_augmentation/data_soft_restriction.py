import numpy as np
import pandas as pd
import itertools

from constants import *


def create_dataset_soft_restricted(gt_csv, modality, spec_str =''):
	# Unlike create_dataset_visual, HNS that do not contain visual queues are made EN instead of dropped

	# If it is multimodal, we want to unify "text+visual" and "visual+test", for instance
	modalities = modality.split("+")
	modalities.sort()
	modality = "+".join(modalities)

	df = pd.read_csv(gt_csv, sep=";", converters={'concepts': pd.eval})
	original_df_name = gt_csv.split('/')[-1][:(-4)]
	new_df_name = f"{original_df_name}_soft_{modality}{spec_str}"

	n_init = len(df)

	visual_concepts = [MODALITY_TO_CONCEPTS[mod] for mod in modalities]
	visual_concepts = list(itertools.chain.from_iterable(visual_concepts))
	some_visual_concept = df['concepts'].apply(lambda clip_concepts: any(x in visual_concepts for x in clip_concepts))
	no_visual_concept = np.bitwise_not(some_visual_concept)

	df_visual = df[some_visual_concept]
	df_non_visual = df[no_visual_concept]
	if spec_str == '_corrected':
		df_non_visual.loc[df_non_visual['label'] != 'Not Sure','label'] = 'Easy Negative'
	else:
		df_non_visual.loc[:, 'label'] = 'Easy Negative'

	final_df = pd.concat([df_visual, df_non_visual])

	n_after = len(final_df)
	n_diff = n_init - n_after

	print(f"Initial length : {n_init}\nLength after removing non-visual 'positive' : {n_after}\n"
		  f"{n_diff} ({round(100 * n_diff / n_init, 2)}% of the original df) examples have been removed.")

	final_df.to_csv(f'{DATASETS_DIR}/{new_df_name}.csv', sep=";", index=False)
	print(f'{DATASETS_DIR}/{new_df_name}.csv created')


def return_dataset_soft_restricted(gt_csv, modality):
	# Unlike create_dataset_visual, HNS that do not contain visual queues are made EN instead of dropped
	# N.B. : this is the corrected version, ie. Not Sure are kept Not Sure

	# If it is multimodal, we want to unify "text+visual" and "visual+test", for instance
	modalities = modality.split("+")
	modalities.sort()

	df = pd.read_csv(gt_csv, sep=";", converters={'concepts': pd.eval, 'start_clips': pd.eval, 'end_clips': pd.eval, 'imdb_keys': pd.eval})

	visual_concepts = [MODALITY_TO_CONCEPTS[mod] for mod in modalities]
	visual_concepts = list(itertools.chain.from_iterable(visual_concepts))
	some_visual_concept = df['concepts'].apply(lambda clip_concepts: any(x in visual_concepts for x in clip_concepts))
	no_visual_concept = np.bitwise_not(some_visual_concept)

	df_visual = df[some_visual_concept]
	df_non_visual = df[no_visual_concept]
	df_non_visual.loc[df_non_visual['label'] != 'Not Sure','label'] = 'Easy Negative'
	final_df = pd.concat([df_visual, df_non_visual])

	return final_df


if __name__ == "__main__":
	gt_csv = f"{DATASETS_DIR}/mobygaze_dataframe.csv"
	for modality in ["visual", "text", "audio", "text+visual", "audio+text+visual"]: #["visual", "text", "audio", "text+visual", "audio+text+visual"]:
		print(f"\n## {modality}:")
		create_dataset_soft_restricted(gt_csv, modality, spec_str='_corrected')
