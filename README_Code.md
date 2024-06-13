# MObyGaze Code

Here are the steps to use our code

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)


## Installation

To install this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/husky-helen/MObyGaze.git
    ```
2. Navigate to the project directory:
    ```sh
    cd MObyGaze
    ```
3. Install the dependencies:
   
   To install the ActionFormerObj code we advise to follow the steps of the original github: https://github.com/happyharrycn/actionformer_release/tree/main. Dependencies are listed in the `Install.md` file.

   For both, LabelDisversity and WSL, you can find a env.txt file listing all the dependencies.

   
   

## Usage

### Preprocessing

In order to use our code,you might need to extract the X-CLIP features. We used a window size of 16 and a stride of 16 as well. 

Also, you might need to do some preprocessing steps on the `dataset/mobygaze_dataframe.csv` file. For example selecting a specific annotator and/or selecting a specific label or grouping 2 labels together
```sh
    import pandas as pd
    df = pd.read_csv("dataset/mobygaze_dataframe.csv", sep=";", index_col=0)

    # annotator selection
    df_annotator = df.query("annotator == 'annotator_1'")

    # label selection
    df_label = df.query("label == 'Easy Negative'")

    # label grouping
    dico_group = {"Easy Negative" : "ENHN", "Hard Negative": "ENHN", "Sure":"Sure"}
    new_labels = df["label"].map(dico_group)
    df["label"] = new_labels
```

For Supervised Learning (SL), Weakly Supervised Learning (WSL), Supervised Audio and Speech Classification, the dataset must be partitioned into distinct folds for the purposes of training, validation, and testing of the models. In the provided code, folds zero, one, and two are designated for testing, validation, and training, respectively. Researchers may modify this column to align with their specific experimental design requirements. Also, its worth to note that a separate dataframe is created for each annotator.

### FSL
1. From the MobyGaze folder go to the LabelDiversity folder:
```sh
    cd FSL
```
2. Change the path in the `process_inference_neurips.py` and `process_train.py` files
3. Train the model
```sh
  python3 train.py
  ```
4. Test the model
 ```sh
  python3 eval.py
  ```

### Label Diversity 
1. From the MobyGaze folder go to the LabelDiversity folder:
```sh
    cd LabelDiversity
```


2. Create your config file according to what LabelDiversity type you want to try in `LabelDiversity/configs/`
3. Train the model (you can check the available arguments in the `LabelDiversity/train.py` file).
  ```sh
  python3 train.py config_file_path
  ```

4. Test the model (you can check the available arguments in the `LabelDiversity/eval.py` file).
```sh
  python3 eval.py config_file_path model_ckpt_path
```

### WSL
1. From the MobyGaze folder go to the WSL folder:
```sh
    cd WSL
```

2. Change the arguments in the `WSL/option.py` file 
3. Train and Test the model (you can also give the appropriate args here)
 ```sh
  python3 main.py
 ```

### ActionFormerObj

1. From the MovyGaze folder go to the ActionFormerObj folder:
```sh
    cd ActionFormerObj
```
2. Use the preprocessing files in the `Preprocessing` folder. In order to replicate our experiments you might need to do some additional preprocessing before using these files. For example create intermediate csv files by selecting a specific annotator or specific labels.
3. Create or modify the config files in `configs`
4. Train the model (you can check the available arguments in the `ActionFormerObj/train.py` file).
 ```sh
  python3 train.py config_file_path
 ```
5. Test the model (you can check the available arguments in the `ActionFormerObj/eval.py` file).
```sh
  python3 eval.py config_file_path model_ckpt_path
```

### AudioClassification

The preparation of the data is detailed in supplementary material of the paper. In order to associate the annotated segments (columns `start_frame` and `end_frame` respectively) with the corresponding span of audio track samples we used `segment_audio.ipynb` located in the `/audio_preprocessing` directory. All the audio segments are saved in a .wav format. In order to run the model:
1. Extract the features from the raw .wav files using the `extract_features.py` file located in  `/extract_features`. Researchers needs to specify path to the annotator csv, the directory of the sannotator's raw audio segmenets and the output directory. The .pt features will be stored in `yourfeaturedir/annotator_name/imdb_key` (e.g yourfeaturedir/annotator_1/tt0108160).
2. Change the path in the `process_train.py` file and specify the type of experiment (var `TYPE_OF_MODEL`)
3. Change the path in the `train.py` file. Researchers need to specify the both path to the extracted features and the one to the data.
4. Train the model
```sh
  python3 train.py
  ```
5. `results_analysis.ipynb` notebook can be used to read from the tensorboard logs. As an alternative, run `tensorboard --logdir=path_to_log_folder`

### SpeechClassification (LLama and DistilRoberta)
The preparation of the data is detailed in supplementary material of the paper. 
1. Change the path in the `process_train.py` file and specify the type of experiment (var `TYPE_OF_MODEL`)
3. Change the paths in the `train.py` file. Researchers need to specify the path to the data.
4. Train the model
```sh
  python3 train.py
  ```

Available script to test LLaMa: <br>
5. Test the model using `eval.py` located in SpeechClassification/LLaMA/. The same script can be easily adapted for DistilRoberta model.
```sh
  python3 eval.py config_file_path model_ckpt_path
```
