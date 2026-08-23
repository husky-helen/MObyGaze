#!/bin/bash

conda deactivate
conda activate vad-r1
nohup python -u XCLIP_extract_clip_features.py > extract_features.out 2> extract_features.err &
