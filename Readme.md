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

The files describing the conda environment used for the various task will be found in each model subfolder as .yml or .txt. The versions specified in this file are the ones that have been tested to work with our code, but other versions may work. Regarding the version of PyTorch that is specified, you may need to change the pytorch version according to your CUDA version. General requirement file is [/models/requirements.txt](https://github.com/husky-helen/MObyGaze/blob/main/models/requirement.txt)

### Citations
If you use this datset, please cite:  
Julie Tores, Elisa Ancarani, Lucile Sassatelli, Hui-Yin Wu, Clement Bergman, Lea Andolfi, Victor Ecrement, Remy Sun, Frederic Precioso, Thierry Devars, Magali Guaresi, Virginie Julliard, and Sarah Lecossais. 2025. MObyGaze: a film dataset of multimodal objectification densely annotated by experts.
arXiv:2505.22084 https://arxiv.org/abs/2505.22084

This dataset has been used in the following articles:  
- Julie Tores, Elisa Ancarani, Remy Sun, Lucile Sassatelli, Hui-Yin Wu, and Frederic Precioso. Re-examining concept-based explainable models for multimodal interpretative tasks. In Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25), pages 12437–12445, New York, NY, USA, 2025. Association for Computing Machinery. doi : 10.1145/3746027.3758170.
- Elisa Ancarani, Julie Tores, Remy Sun, Lucile Sassatelli, Hui-Yin Wu, and Frederic Precioso. Leveraging concept annotations for trustworthy multimodal video interpretation through modality specialization. In Proceedings of the 1st International Workshop on Cognition-oriented Multimodal Affective and Empathetic Computing, pages 11–19, 2025.
- Elisa Ancarani, Julie Tores, Lucile Sassatelli, Rémy Sun, Hui-Yin Wu, Frédéric Precioso. 2025. Leveraging Multimodal Explanatory Annotations for Video Interpretation with Modality Specific Dataset. arxiv:2504.11232.
- Julie Tores, Lucile Sassatelli, Hui-Yin Wu, Clement Bergman, Lea Andolfi, Victor Ecrement, Frederic Precioso, Thierry Devars, Magali Guaresi, Virginie Julliard, and Sarah Lecossais. Visual objectification in films : Towards a new AI task for video interpretation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10864–10874, Seattle, WA, USA, 2024a. IEEE. doi : 10.1109/CVPR52733.2024.01033.
