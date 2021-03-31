# Directory Structure


## examples
- basic code to visualize images and state information
- STEP 0: `python3 examples/load_initial_dataset.py` 
    - creates a few visualized images in the `scratch` subfolder
    - do not check results from `scratch` into GIT
    - `scratch/viz` shows camera images from a few runs with state information
- STEP 1: `python3 examples/image_dataloader.py`
    - sample pytorch dataloader to load images and corresponding state variables

## train DNN
- code to train an LEC for vision to estimate distance to centerline and other state information for an airplane in X-Plane

## utils
- generic utilities

