import os
import pandas as pd

from global_utils import str_conversion_from_exp_name, classif_info_from_exp_name
from constants import *

base_dataset_name = "mobygaze_dataframe_soft_visual_corrected"
exp_name = 'EN_HNS'

csv_path = f"{DATA_DIR}/{base_dataset_name}.csv"
base_df = pd.read_csv(csv_path, sep=";", converters={'concepts': pd.eval})

# 1. We create the column label_int
classif_info = classif_info_from_exp_name(exp_name)
type_classif = classif_info["type_classif"]  # either "label" or "concept" for now
classif_classes = classif_info["classif_classes"]
if type_classif == 'label':
	dico_level = classif_classes
	# create the label_int column and filter out labels not in dico_level
	base_df['label_int'] = base_df['label'].map(dico_level)
	base_df = base_df.dropna(subset=['label_int'])
	# convert label_int to integer type
	base_df['label_int'] = base_df['label_int'].astype(int)
elif type_classif == 'concept':
	positive_classes = classif_classes
	base_df['label_int'] = base_df['concepts'].apply(
		lambda clip_concepts: int(any(x in positive_classes for x in clip_concepts)))
else:
	raise Exception(f"Unknown classif: {type_classif}")

# 2. We create the column label_str
str_conversion = str_conversion_from_exp_name(exp_name)
int_conversion = {v: k for k, v in str_conversion.items()}
base_df["label_str"] = base_df["label_int"].map(int_conversion)

# 3. We save the newly obtained df
csv_directory = f"{PROJECT_DIR}/data/demo_pool_from_base"
if not os.path.exists(csv_directory):
	os.makedirs(csv_directory)
combined_csv_path = f"{csv_directory}/{base_dataset_name}_demo_pool_for_{exp_name}.csv"
base_df.to_csv(combined_csv_path, sep=';', index=False)
print(f"\nSaved successfully demonstration pool dataset at {combined_csv_path}")
