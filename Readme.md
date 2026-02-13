# MObyGaze
This repository stores the dataset and code introduced in the submitted article *MObyGaze: a film dataset of multimodal objectification densely annotated by experts*.
The MObyGaze dataset is described in the Croissant format for machine learning dataset, in the metadata.json file. This file specifies the resource file descriptions and data structure. 

## Repository Structure

### Dataset 
The dataset, employed for the training and evaluation of our models, is made of four files described in the metadata.json file under the Croissant format. The resource files are available under the `/dataset` directory. Within this directory, four files are included:

1. **mobygaze_movielist.csv**: contains the complete list of the movies with detailed information.
2. **objectification-thesaurus.json**: details the possibles levels of objectification to annotate, as well as the list of concepts and their respective instances.
3. **mobygaze_dataframe.csv**: the entire table of annotated segments, where a segment corresponds to an interval of a movie delimited by a given annotator. The models were trained using this file, with necessary adaptations made according to the specific task requirements. A description of this datasets can be found in the Readme_data.md
4. **mobygaze_db.sql**: the sql description of the dataset, containing the above resources as tables for movies, annotations, and subtitles.
   
### Models

All the models were implemented using PyTorch. They will be shared in folder `/models`.

### Requirements

The files describing the conda environment used for the various task can be found in each project folder as .yml or .txt (e.g `/FSL/fsl.yml`).  The versions specified in this file are the ones that have been tested to work with our code, but other versions may work. Regarding the version of PyTorch that is specified, you may need to change the pytorch version according to your CUDA version. General requirement file is [requirements.txt](https://github.com/husky-helen/MObyGaze/blob/main/requirement.txt)
