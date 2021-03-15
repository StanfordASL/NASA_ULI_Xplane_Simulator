"""
    Goal: Visualize images from aircraft camera and load as a pytorch dataloader

    1. load all example png examples as a pytorch dataloader
    2. save a few images to disk for visualization
    3. load the corresponding state information in the h5 file
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
    data_dir = DATA_DIR + '/test_dataset/'

    label_file = data_dir + '/labels.h5'
    
    with h5py.File(label_file, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        for i in range(len(f.keys())):
            a_group_key = list(f.keys())[i]
            print('key: ', a_group_key)
            # Get the data
            data = list(f[a_group_key])
            print('value: ', data)



