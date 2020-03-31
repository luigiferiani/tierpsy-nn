# Recognize worms

This part of the repo is about training a neural network that can recognise which segmented objects are worms and which ones are not.
Only actual worms should then be skeletonised in Tierpsy.

#### Caveat

This readme file was created after I went through the code, but I am not the original developer so may be missing some points or may be inaccurate.
This readme is also only on the few files in this folder that I've gone through.

## Flow

1. Create a sample set for manual annotation
2. Manually annotate the sample set using the GUI
3. Reformat the manually annotated sample dataset for training
4. Train a model?


### 1. Create a sample set for manual annotation
The first step is to use masked videos to create a sample for manual annotation and neural network training, using `./keras/get_sample_from_video.py`
This script will loop through all the masked videos in a folder that you'll need to specify in the code (variable `root_dir`), extract ROIs (regions of interest) where Tierpsy thinks there may be a worm, and write them in an `hdf5` file the location of it is written in the `all_samples_file` variable.

You could, if you wanted to, customise every how many frames you want to extract ROIs from the masked videos. The default is to use the frames that were also saved in full_data in the masked videos.

ROIs are by default `160`x`160` px.

### 2. Manually annotate the sample set using the GUI
The sample dataset created at the previous step can be manually annotated using a GUI.
The GUI is started by executing `./image_labeler/image_labeler.py`.

##### Selecting the sample dataset
You can either point the GUI to the dataset you created in the previous step by hardcoding its path into the variable `FILE_NAME` at the very beginning of `./image_labeler/image_labeler.py`, or you can use the GUI to select the file manually (either by clicking on `Select Video File` or just drag-and-drop).

##### Marking worms
The GUI will present you with a ROI of a worm. You can mark what the worm is by clicking on the corresponding button, just below the main window in the GUI.
Alternatively, you can use the keyboard (just press the number corresponding to the label you want to assign to the ROI):
1. Not a worm
2. Valid worm
3. Difficult worm _Note: I do not know the definition of "difficult"_
4. Worms aggregate
5. Eggs
6. Larvae

After assigning a label to a ROI, you can progress to the following one. You can move to the following ROI by pressing `>`, or `.`, on your keyboard. You can return to a previous ROI by pressing `<` or `,`. You can also move to a particular ROI by writing a number in the `sample number` field.
If you move to a ROI that has been already labelled, you'll be able to review its status as the relevant button will be highlighted. This makes for very quick review of manual annotations.

##### Saving your work
As you label ROIs, the annotations are only saved in the RAM of the computer. If the program were to crash, the annotations would in all likelihood be lost.
By clicking on `SAVE`, the annotations are stored on disk, in the sample dataset `hdf5` file.

The GUI will prompt you to save when you try to close it, but it's better practice to save much more often than that.

#### Structure of the sample dataset
By inspection of the sample dataset via HDFView, this is what you can find inside:
1. `file_list`

    It's an `N`-by-`2` dataframe with columns `file_id` and `file_name`. It lists all the masked videos the ROIs come from.

2. `sample_data`

    This is the most important dataframe in the file.
    It has `M` rows, where M is the number of ROIs in the sample file. It contains all the ROI-specific information:

    * `frame_number`: frame in the original video this ROI was taken from.
    * `worm_index_joined`
    * `skeleton_id`
    * `coord_x`
    * `coord_y`
    * `threshold`
    * `has_skeleton`
    * `roi_size`: size of the original ROI as per Tierpsy. Here all ROIs are of the size specified in `get_sample_from_videos.py`
    * `area`
    * `timestamp_raw`
    * `timestamp_time`
    * `is_good_skel`
    * `img_row_id`: index to which slice in `full_data` and `mask` contains this ROI
    * `file_id`: matches the homonymous column in `file_list` - which file does the ROI come from
    * `roi_corner_x`
    * `roi_corner_y`
    * `resampled_index`: `sample number` in the GUI, it's just the order in which the GUI shows the ROIs to the user. Only appears after the first time the file is opened in the GUI.
    * `label_id`: manually assigned label. If `0`, this ROI was not manually annotated. Only appears after the first time the file is opened in the GUI.


3. `full_data`

    This is an `M`-by-`ROI_size`-by-`ROI_size` array. Each slice is a ROI. These ROIs are taken from the `/full_data` node in the masked videos (i.e. with no masking applied).

4. `mask`

    This is an `M`-by-`ROI_size`-by-`ROI_size` array. Each slice is a ROI. These ROIs are taken from the `/mask` node in the masked videos (i.e. masking applied).

5. `contour_side1`, `contour_side2`, `skeleton`

    These are all `M`-by-`49`-by-`2` arrays, and store the coordinates of the candidates for contour and skeleton of the worms.


### 3. Reformat the manually annotated sample dataset for training
This should happen by running the script at `./keras/reformat_for_sample.py`.
To a very quick read, the script reads the manually annotated file specified by the variable `all_samples_file`, takes the subset of ROIs that were **actually** labelled, and divides them into `train`, `test`, and `val` datasets.
These are saved in the file specified by `output_samples_file`.

By default, the test and validation sets are made of `1000` ROIs each.
Again by default, only the central `80`x`80` px region of each labelled ROI is saved in the reformatted `output_samples_file` (so the model is trained on these `80`x`80` px images).
