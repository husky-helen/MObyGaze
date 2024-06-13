# MObyGaze

## Repository Structure

### Datasets 
The dataset employed for the training and evaluation of our models is available within this repository, specifically located in the `/dataset` directory. Within this directory, four files are included:

1. **Neurips.sql**: is the primary source of the data, containing detailed information regarding the movies, their annotations, and subtitles.
2. **mobygaze_dataframe.csv**: represents the SQL contents, including the target column relevant to the specific task at hand. The models were trained using this file, with necessary adaptations made according to the specific task requirements.
3. **mobygaze_movielist.csv**: contains the complete list of the movies.
4. **objectification-thesaurus.json**: details the various annotation choices, the types of concepts, and their respective explanations.

### Models

All the models where implemented using PyTorch. 

All the code is organised into six distinct directories: `ActionFormerObj` , `AudioClassification`, `FSL`, `LabelDiversity`, `SpeechClassification`, `WSL`. Each directory corresponds to a specific task, allowing independent execution of the code within these folders. Further details on the content of this folder on how to run the code can be found in the README_code.md


### Requirements

The files describing the conda environment used for the various task can be found in each project folder as .yml or .txt (e.g `/FSL/fsl.yml`).  The versions specified in this file are the ones that have been tested to work with our code, but other versions may work. Regarding the version of PyTorch that is specified, you may need to change the pytorch version according to your CUDA version. General requirement file is [requirements.txt](https://github.com/husky-helen/MObyGaze/blob/main/requirement.txt)
