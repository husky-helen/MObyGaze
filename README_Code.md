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
   **TODO add the env.yaml file for each code folder**
   

## Usage

### Label Diversity 
0. From the MobyGaze folder go to the LabelDiversity folder:
```sh
    cd LabelDiversity
```
**TODO : rename disagreement to LabelDiversity**
Here's a quick example of how to use the project:

1. Create your config file according to what LabelDiversity type you want to try in `LabelDiversity/configs/`
2. Train the model (you can check the available arguments in the `LabelDiversity/train.py` file).
  ```sh
  python3 train.py config_file_path
  ```

3. Test the model.
```sh
  python3 eval.py config_file_path model_ckpt_path
```


### WSL
0. From the MobyGaze folder go to the WSL folder:
```sh
    cd WSL
```

1. Change the arguments in the `WSL/option.py` file 
2. Train and Test the model (you can also give the appropriate args here)
 ```sh
  python3 main.py
 ```
