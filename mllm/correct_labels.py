import numpy as np
from global_utils import *
import os
import pandas as pd


def update_labels_in_csvs(directory, correct_labels_df):
	converters = {
		'concepts': pd.eval,
		'start_clips': pd.eval,
		'end_clips': pd.eval,
		'imdb_keys': pd.eval
	}

	# Create a fast lookup: guid -> label
	label_map = correct_labels_df.set_index("guid")["label"]

	for root, _, files in os.walk(directory):
		for file in files:
			if not file.endswith(".csv"):
				continue

			csv_path = os.path.join(root, file)
			print(f"Processing {csv_path}")

			# Read the CSV
			df = pd.read_csv(
				csv_path,
				sep=";",
				converters=converters
			)

			if "guid" not in df.columns or "label" not in df.columns:
				print(f"Skipping {csv_path}: missing guid or label column")
				continue

			# Replace labels where guid exists in correct_labels_df
			df["label"] = df["guid"].map(label_map).combine_first(df["label"])

			# Save in place
			df.to_csv(csv_path, sep=";", index=False)


for modality in ['text', 'text+visual']:
	gt_csv = f"{PROJECT_DIR}/data/mobygaze_dataframe.csv"
	correct_labels_df = return_dataset_soft_restricted(gt_csv, modality)
	directory = f"{PROJECT_DIR}/inference_results/EN_HNS_final/zero_shot/Qwen2.5-VL-7B-Instruct/mobygaze_dataframe_soft_visual_corrected_{modality}/"
	update_labels_in_csvs(directory, correct_labels_df)
