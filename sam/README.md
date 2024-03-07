Note: hyperparameters of sam feature are located under cfg.sam section

Usage Instruction:

* download sam model from https://github.com/facebookresearch/segment-anything
* put local path to it inside your config
* put your images and their json annotations inside config.sam.root_path/data/original directory
* run sam_run.py script under root to get annotations

This feature has functionality to keep your temporary calculation results inside "auxiliary" folder. 
If you are running it for the 1st time, or you're changing any of following hyperparameters:
* sam_model_name
* min_mask_region_area
* threshold_orig_overlap
* min_polygon_area
* epsilon
* any data inside your original json annotations

then new sam inference will start. Otherwise, sam inference results from previous iteration will be taken.

Hyperparameters descriptions are commented inside config file.
