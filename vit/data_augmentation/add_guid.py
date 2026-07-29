import pandas as pd

from constants import *


# This script should be run before anything else
gt_csv = f"{DATASETS_DIR}/mobygaze_dataframe.csv"

original_df = pd.read_csv(gt_csv, sep=';', converters={'concepts': pd.eval})
if "guid" not in original_df.columns:
	original_df.loc[:, 'guid'] = original_df.index
# We store it back with the modifications
original_df.to_csv(gt_csv, sep=';', index=False)