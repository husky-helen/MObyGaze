import os


# Directories
PROJECT_DIR = f""  # TODO
DATASETS_DIR = f"{PROJECT_DIR}/datasets"
CHECKPOINTS_DIR = ""  # TODO
# models checkpoints are .gitignored anyway
TATOOVI_DIR = ""  # TODO
# TATOOVI_DIR is expected to contain one folder "[imdb]-[title]" for each movie, containing the .json file

# Our features dir are designed for Transformers model,
# so we choose the "sequential version" of the features, not the pooled one
FEATURE_DIR = ""  # TODO

VISUAL_FEATURE_DIR = ""  # TODO
TEXT_FEATURE_DIR = ""  # TODO
AUDIO_FEATURE_DIR = ""  # TODO

# Partition of data for cross-validation

FOLDS_INFO = {
	0: {"test": ["tt0097576", "tt0822832", "tt0108160", "tt0970416"], "validation": ["tt0109830", "tt1142988"]},
	1: {"test": ["tt1454029", "tt1568346", "tt0119822", "tt1907668"], "validation": ["tt2267998", "tt0375679"]},
	2: {"test": ["tt1285016", "tt2267998", "tt1193138", "tt0375679"], "validation": ["tt0110912", "tt1045658"]},
	3: {"test": ["tt0467406", "tt0109830", "tt1570728", "tt1142988"], "validation": ["tt1285016", "tt1193138"]},
	4: {"test": ["tt0110912", "tt0120338", "tt1045658", "tt1632708"], "validation": ["tt0097576", "tt0108160"]}
}
PSEUDO_LABELING_FOLDS_INFO = {
	0: {"test": ["tt0097576", "tt0822832", "tt0108160", "tt0970416"], "validation": ["tt0109830", "tt1142988", "tt1454029", "tt1568346"]},
	1: {"test": ["tt1454029", "tt1568346", "tt0119822", "tt1907668"], "validation": ["tt2267998", "tt0375679", "tt0822832", "tt0970416"]},
	2: {"test": ["tt1285016", "tt2267998", "tt1193138", "tt0375679"], "validation": ["tt0110912", "tt1045658", "tt0119822", "tt1907668"]},
	3: {"test": ["tt0467406", "tt0109830", "tt1570728", "tt1142988"], "validation": ["tt1285016", "tt1193138", "tt0120338", "tt1632708"]},
	4: {"test": ["tt0110912", "tt0120338", "tt1045658", "tt1632708"], "validation": ["tt0097576", "tt0108160", "tt0467406", "tt1570728"]}
}
DIRECT_APPROACH_PSEUDO_LABELING_FOLDS_INFO = {
	0: {"test": ["tt0097576", "tt0822832", "tt0108160", "tt0970416"], "validation": []},
	1: {"test": ["tt1454029", "tt1568346", "tt0119822", "tt1907668"], "validation": []},
	2: {"test": ["tt1285016", "tt2267998", "tt1193138", "tt0375679"], "validation": []},
	3: {"test": ["tt0467406", "tt0109830", "tt1570728", "tt1142988"], "validation": []},
	4: {"test": ["tt0110912", "tt0120338", "tt1045658", "tt1632708"], "validation": []}
}
NUM_FOLDS = [0, 1, 2, 3, 4]

ALL_MOVIES = ["tt0097576", "tt0822832", "tt0108160", "tt0970416", "tt1454029", "tt1568346", "tt0119822", "tt1907668",
			  "tt1285016", "tt2267998", "tt1193138", "tt0375679", "tt0467406", "tt0109830", "tt1570728", "tt1142988",
			  "tt0110912", "tt0120338", "tt1045658", "tt1632708"]

# Concepts

VISUAL_CONCEPTS = ['Type of shot', 'Look', 'Body', 'Posture', 'Clothing', 'Appearance', 'Exp of emotion', 'Activities']
ALL_CONCEPTS = ['Type of shot', 'Look', 'Body', 'Posture', 'Clothing', 'Appearance', 'Exp of emotion', 'Activities',
				'Speech', 'Voice', 'Soundtrack']
NON_VISUAL_CONCEPTS = ['Speech', 'Voice', 'Soundtrack']
TEXT_CONCEPTS = ['Speech']
AUDIO_CONCEPTS = ['Voice', 'Soundtrack']
MODALITY_TO_CONCEPTS = {'visual': VISUAL_CONCEPTS, 'text': TEXT_CONCEPTS, 'audio': AUDIO_CONCEPTS,
						'text+visual': TEXT_CONCEPTS + VISUAL_CONCEPTS, 'audio+text': AUDIO_CONCEPTS + TEXT_CONCEPTS,
						'audio+visual': AUDIO_CONCEPTS + VISUAL_CONCEPTS,
						'audio+text+visual': AUDIO_CONCEPTS + TEXT_CONCEPTS + VISUAL_CONCEPTS}

TARGET_CONCEPTS = ['Type of shot', 'Body', 'Posture', 'Clothing']  # the concepts of special interest
