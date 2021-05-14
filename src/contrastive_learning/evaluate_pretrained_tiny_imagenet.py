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

import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

DATA_DIR = NASA_ULI_ROOT_DIR + 'data/'

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

NNET_UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/simulation/'
sys.path.append(NNET_UTILS_DIR)
from nnet import *

from textfile_utils import *

if __name__ == '__main__':

    prefix = 'evaluate_tiny_taxinet'
    train_test_split_list = ['train', 'validation', 'test']

    MAX_IMAGES = 10
    NROW = 5

    # Read in the pretrained tiny taxinet DNN
    filename = "../../models/TinyTaxiNet.nnet"
    network = NNet(filename)

    condition_list = ['afternoon', 'morning', 'night', 'overcast']

    for condition in condition_list:
        # create a temp dir to visualize a few images
        visualization_dir = SCRATCH_DIR + '/' + prefix + '/' + condition
        remove_and_create_dir(visualization_dir)

        for train_test_split in train_test_split_list:
            
            print('evaluating: ', condition, train_test_split)

            # where original XPLANE images are stored 
            label_file = DATA_DIR + 'downsampled/' + condition + '/' + '_'.join([condition, train_test_split, 'downsampled']) + '.h5'

            f = h5py.File(label_file, "r")
            # get the training data
            x_train = f['X_train'].value
            y_train = f['y_train'].value
           
            print(' ')
            print('X_train shape: ', x_train.shape)
            print('y_train shape: ', y_train.shape)
            print(' ')

            # plot vector of images
            ####################
            num_images_total = x_train.shape[0]

            start_index = np.random.randint(num_images_total - MAX_IMAGES)
            start_index = 0

            x_array_np = x_train[start_index: start_index + MAX_IMAGES]
            y_array_np = y_train[start_index: start_index + MAX_IMAGES]

            # evaluate the DNN for a single input
            for idx in range(x_array_np.shape[0]):
                pred = network.evaluate_network(x_array_np[idx].flatten())
                target = y_array_np[idx]
                target_pred = target[0:2]
                error = np.linalg.norm(target_pred - pred)
                print(' ')
                print('idx: ', idx)
                print('pred: ', pred)
                print('target_pred: ', target_pred)
                print('error: ', error)
                print(' ')

            f.close()
