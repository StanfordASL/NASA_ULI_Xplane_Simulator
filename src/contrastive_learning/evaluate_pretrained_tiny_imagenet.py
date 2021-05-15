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

    # 
    df_condition_list = []
    df_train_test_list = []
    df_target_pred_list = []
    df_pred_list = []
    df_error_list = []

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

            # to get a subset of data
            start_index = np.random.randint(num_images_total - MAX_IMAGES)
            start_index = 0

            x_array_np = x_train[start_index: start_index + MAX_IMAGES]
            y_array_np = y_train[start_index: start_index + MAX_IMAGES]

            # to use all data
            x_array_np = x_train
            y_array_np = y_train

            # evaluate the DNN for a single input
            for idx in range(x_array_np.shape[0]):
                pred = network.evaluate_network(x_array_np[idx].flatten())
                target = y_array_np[idx]
                target_pred = target[0:2]
                error = np.linalg.norm(target_pred - pred)

                #print(' ')
                #print('idx: ', idx)
                #print('target_pred: ', target_pred)
                #print('pred: ', pred)
                #print('error: ', error)
                #print(' ')

                # add data to a list
                df_condition_list.append(condition)
                df_train_test_list.append(train_test_split)
                df_target_pred_list.append(target_pred)
                df_pred_list.append(pred)
                df_error_list.append(error)

            f.close()

    df = pandas.DataFrame({'Condition': df_condition_list, 'Train/Test': df_train_test_list, 'Error': df_error_list}) 
    
    csv_fname = SCRATCH_DIR + '/' + prefix + '/results.csv'
    df.to_csv(csv_fname)

