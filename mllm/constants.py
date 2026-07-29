import os

# If you run the code on a computer-like cluster :
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so that gpu index matches nvidia-smi index
#os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_GPU  # "0,1" for instance

PROJECT_DIR = ""  # TODO
MOVIE_DIR = f""  # TODO

# (Sequential) features directories
VISUAL_FEATURE_DIR = ""  # TODO
NEW_VISUAL_FEATURE_DIR = ""  # TODO
TEXT_FEATURE_DIR = ""  # TODO
AUDIO_FEATURE_DIR = ""  # TODO
# Features extracted from ViT (N.B. : not used, except in few-shot for feature_extractor="vit")
VIT_FEATURE_DIR = ""  # TODO

PROMPT_DIR = f"{PROJECT_DIR}/prompts"
DATA_DIR = f"{PROJECT_DIR}/data"

IMDB_TO_MOVIE_DIR = {directory.split("-")[0]: directory for directory in os.listdir(MOVIE_DIR)}
IMDB_TO_VIDEO_FILE = {directory.split("-")[0] : next(x for x in os.listdir(f"{MOVIE_DIR}/{directory}")
													 if x.startswith(f"{directory}."))
					  for directory in os.listdir(MOVIE_DIR)}
# We need this slightly elaborate code because the video may be in various formats
# N.B. : For this code to work, we expect names of videos to match the folder they are in

# Movies and folds
# Since there is no proper training, there is no proper validation neither
# We set validation to [], so that the complementary of test+val, which is test, does correspond to all other test folds
FOLDS_INFO = {
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

# Prompting
OLD_SYSTEM_PROMPT = ("You are an expert film analyst specialized in identifying the objectification of characters in cinema. "
				 "When given a clip, or a speech, you must examine the ENTIRE clip, not only its beginning or end -- "
				 "relevant cues can appear at any point and everything must be considered before concluding. "
				 "Ground every claim strictly in what is actually shown or written -- never invent detail that is not there.")

SYSTEM_PROMPT = (
    "You are an expert film analyst specializing in identifying the objectification "
    "of characters in cinema. Objectification means portraying a character primarily "
    "as a physical or sexual object, or as passive/subordinate, rather than as a full "
    "subject with agency and interiority.\n\n"
    "You will be shown a video clip in full. You must examine the ENTIRE clip, not "
    "only its beginning or end -- relevant cues can appear at any point and all must "
    "be considered before concluding. An instance counts as present even if it "
    "appears in just a single frame or a brief moment of the clip -- a fleeting or "
    "momentary occurrence is still valid evidence and must not be dismissed just "
    "because it is not sustained throughout the clip.\n\n")

# For Intern pre-processing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
