"""
    Goal: Visualize images from aircraft camera and load as a pytorch dataloader

    load the state information saved as a csv file

"""

import sys, os
import torch
import numpy as np
import pandas
import matplotlib.pyplot as plt
import h5py

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']

import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

DATA_DIR = NASA_ULI_ROOT_DIR + '/data/'

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *

if __name__ == '__main__':
    # how often to plot a few images for progress report
    # warning: plotting is slow
    NUM_PRINT = 5

    # create a temp dir to visualize a few images
    visualization_dir = SCRATCH_DIR + '/viz/'

    # where the final dataloader will be saved
    DATALOADER_DIR = SCRATCH_DIR + '/dataloader/'

    # where original XPLANE images are stored 
    data_dir = DATA_DIR + '/test_dataset_smaller_ims/'

    label_file = data_dir + '/labels.csv'
   
    df = pandas.read_csv(label_file, sep=',')

    # df is a dataframe of images and their corresponding state information

    # ['image_filename', 'absolute_time_GMT_seconds', 'relative_time_seconds', 'distance_to_centerline_meters', 'distance_to_centerline_NORMALIZED', 'downtrack_position_meters', 'downtrack_position_NORMALIZED', 'heading_error_degrees', 'heading_error_NORMALIZED', 'period_of_day', 'cloud_type']

