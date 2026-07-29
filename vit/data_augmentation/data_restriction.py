import pandas as pd

from constants import *


def create_dataset_restricted(gt_csv, modality):
	# Unlike create_dataset_soft_visual, HNS that do not contain visual queues are dropped instead of made EN

	df = pd.read_csv(gt_csv, sep=";", converters={'concepts': pd.eval})
	original_df_name = gt_csv.split('/')[-1][:(-4)]
	new_df_name = f"{original_df_name}_{modality}"

	n_init = len(df)

	visual_concepts = MODALITY_TO_CONCEPTS[modality]
	some_visual_concept = df['concepts'].apply(lambda clip_concepts: any(x in visual_concepts for x in clip_concepts))
	easy_negative = df['label'].apply(lambda label: label == 'Easy Negative')
	kept_indices = easy_negative | some_visual_concept

	df = df[kept_indices]
	n_after = len(df)
	n_diff = n_init - n_after

	print(f"Initial length : {n_init}\nLength after removing non-visual 'positive' : {n_after}\n"
		  f"{n_diff} ({round(100 * n_diff / n_init, 2)}% of the original df) examples have been removed.")

	df.to_csv(f'{DATASETS_DIR}/{new_df_name}.csv', sep=";", index=False)
	print(f'{DATASETS_DIR}/{new_df_name}.csv created')


if __name__ == "__main__":
	gt_csv = f"{DATASETS_DIR}/mobygaze_dataframe.csv"
	for modality in ["visual", "text", "audio"]:
		print(f"\n## {modality}:")
		create_dataset_restricted(gt_csv, modality)
