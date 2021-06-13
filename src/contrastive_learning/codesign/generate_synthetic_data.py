"""
"""

import sys, os
import torch
import numpy as np
import pandas
import matplotlib.pyplot as plt
import h5py
import torchvision

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']

from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

DATA_DIR = NASA_ULI_ROOT_DIR + 'data/'

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *

def create_synthetic_perception_training_data(x_target = 0, max_x = 10, num_samples = 1000, bias = 0, noise_sigma = 0.5, print_mode = True):

    x_lims = [x_target, x_target + max_x]

    # x_robot, x_target, p_true, p_noisy
    num_cols = 4 
    data_matrix = np.zeros([num_samples, num_cols])

    for i in range(num_samples):
       
        # sample x_robot, ensure it is always greater than x_target
        x_robot = np.random.uniform(x_lims[0], x_lims[1])

        p_true = x_robot - x_target

        p_noisy = np.random.normal(p_true + bias, scale=noise_sigma)

        data_vector = [x_robot, x_target, p_true, p_noisy]

        if print_mode:
            print(data_vector)

        data_matrix[i,:] = data_vector

    return data_matrix

if __name__ == '__main__':

    data_matrix = create_synthetic_perception_training_data(x_target = 0, max_x = 10, num_samples = 100, bias = 0, noise_sigma = 0.5, print_mode = True)


