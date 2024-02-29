Usage Instruction:

I am reverencing to "sam" section under config.yml when I talk about hyperparameters

Put your images and their json annotations inside config.sam.root_path/data/original directory
This feature has functionality to keep your temporary calculation results. 
If you are changing following hyperparameters:
* anything under "embeddings_model"
* anything under "data.preprocessing"
* data.n_clusters
then sam inference results from previous iteration (if it exists) will be taken and clustering part only will be started

Take into account, that in case if
* it is your first run
* you change current or add new json files in original folder
* you change any of these hyperparameters:
    ** sam_model_name
    ** min_mask_region_area
    ** threshold_orig_overlap
    ** min_polygon_area
    ** epsilon
then sam inference will be started from the very beginning
