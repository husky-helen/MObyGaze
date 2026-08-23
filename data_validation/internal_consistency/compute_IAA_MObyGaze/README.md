This folder implements the method of  
> Braylan Alexander, Alonso Omar, Lease Matthew. Measuring Annotator Agreement Generally across Complex Structured, Multi-object, and Free-text Annotation Tasks. Proceedings of the ACM Web Conference 2022.

to compute inter-annotator agreement (IAA) on the levels of objectification and concepts of the MObyGaze dataset.
Adaptations have been made for MObyGaze data and a distance function named dist@IoUthresh has been added (see the MObyGaze DMLR article referenced in the main README of this repository).

The files modified from the orginial repository [https://github.com/Praznat/annotationmodeling] are:
- experiment_manager.py
- experiments.py
- agreement.py
- eval_functions.py
- utils.py

## Usage  
From experiment_manager_MObyGaze.py, adapt path to data in line 273 then modify main to run
```
test_MObyGaze(distance_choice="evalSimAtIoU")
```
or
```
test_MObyGaze(distance_choice="eval_f1")
```

