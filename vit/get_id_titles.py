import pandas as pd
import json

from constants import *

gt_csv = f"{DATASETS_DIR}/mobygaze_dataframe_visual.csv"
df = pd.read_csv(gt_csv, sep=";", converters={'concepts': pd.eval})
id_to_title = dict()

for id in df["movie_id"].unique():
	id_to_title[str(int(id))] = df.query("movie_id ==@id")["movie_title"].iloc[0]
# With json the keys are stored as strings anyway ; this way we don't forget it

print(id_to_title)

with open(f'{PROJECT_DIR}/util_data/id_to_title.json', 'w') as file:
	json.dump(id_to_title, file, indent=4)

# with open(f'{PROJECT_DIR}/util_data/id_to_title.json', 'r') as file:
# id_to_title = json.load(file)
