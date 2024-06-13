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

Also, you might need to do some preprocessing steps on the `dataset/mobygaze_dataframe.csv` file. For exemple selecting a specific annotator and/or selecting a specific label or grouping 2 labels together
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
  ``

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
2. Use the preprocessing files in the `Preprocessing` folder. In order to replicate our experiments you might need to do some additional preprocessing before using these files. For exemple create intermediate csv files by selecting a specific annotator or specific labels.
3. Create or modify the config files in `configs`
4. Train the model (you can check the available arguments in the `ActionFormerObj/train.py` file).
 ```sh
  python3 train.py config_file_path
 ```
5. Test the model (you can check the available arguments in the `ActionFormerObj/eval.py` file).
```sh
  python3 eval.py config_file_path model_ckpt_path
```
