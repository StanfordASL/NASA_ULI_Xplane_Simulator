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

def create_synthetic_perception_training_data(x_target = 0, min_x = 5, max_x = 15, num_samples = 1000, bias = 0, noise_sigma = 0.5, print_mode = True, params = None, dtype=torch.float32, v_max = 3):

    x_lims = [min_x, max_x]

    # x_robot, x_target, p_true, p_noisy
    num_features = 2
    data_matrix_x = np.zeros([num_samples, num_features])
    data_matrix_y = np.zeros([num_samples, 1])
    data_matrix_y_true = np.zeros([num_samples, 1])
    data_matrix_v_true = np.zeros([num_samples, 1])

    for i in range(num_samples):
      
        v_robot = np.random.uniform(0, v_max)

        # sample x_robot, ensure it is always greater than x_target
        x_robot = np.random.uniform(x_lims[0], x_lims[1])

        p_true = x_robot - x_target

        p_noisy = np.random.normal(p_true + bias, scale=noise_sigma)

        data_vector_x = [x_robot, x_target]
        data_vector_y = [p_noisy]
        data_vector_y_true = [p_true]
        data_vector_v_true = [v_robot]

        if print_mode:
            print(data_vector)

        data_matrix_x[i,:] = data_vector_x
        data_matrix_y[i,:] = data_vector_y
        data_matrix_y_true[i,:] = data_vector_y_true
        data_matrix_v_true[i,:] = data_vector_v_true

    tensor_dataset = torch.utils.data.TensorDataset(torch.tensor(data_matrix_x, dtype=dtype), torch.tensor(data_matrix_y, dtype=dtype), torch.tensor(data_matrix_y_true, dtype=dtype), torch.tensor(data_matrix_v_true, dtype=dtype))

    tensor_dataloader = torch.utils.data.DataLoader(tensor_dataset, **params)

    return tensor_dataset, tensor_dataloader

if __name__ == '__main__':

    params = {'batch_size': 1,
		  'shuffle': False,
		  'num_workers': 1}

    tensor_dataset, tensor_dataloader = create_synthetic_perception_training_data(x_target = 0, max_x = 10, num_samples = 100, bias = 1.0, noise_sigma = 0.5, print_mode = False, params = params)

    for i, (x_vector, p_noisy , p_true, v_true) in enumerate(tensor_dataloader):
        print('x: ', x_vector)
        print('y_noisy: ', p_noisy)
        print('y_true: ', p_true)
        print('v_true: ', v_true)


