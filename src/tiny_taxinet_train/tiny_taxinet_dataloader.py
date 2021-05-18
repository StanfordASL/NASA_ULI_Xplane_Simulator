import sys, os
import torch
import numpy as np
import pandas
import matplotlib.pyplot as plt
import h5py
import torchvision

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']

DATA_DIR=os.environ['NASA_DATA_DIR']

from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *

def tiny_taxinet_prepare_dataloader(DATA_DIR, condition_list, train_test_split, batch_params, prefix='downsampled', print_mode = True):

    for condition in condition_list:
        # where original XPLANE images are stored 
        label_file = DATA_DIR + 'downsampled/' + condition + '/' + '_'.join([condition, train_test_split, 'downsampled']) + '.h5'


        f = h5py.File(label_file, "r")
        # get the training data, plot a few random images
        x_train = f['X_train'].value
        y_train = f['y_train'].value
    
        if print_mode:
            print(' ')
            print('condition: ', condition, ', train_test_split: ', train_test_split)
            print('X_train shape: ', x_train.shape)
            print('y_train shape: ', y_train.shape)
            print(' ')

    # tensor dataset
    tensor_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))

    tensor_dataloader = torch.utils.data.DataLoader(tensor_dataset, **params)

    return tensor_dataset, tensor_dataloader


if __name__ == '__main__':

    params = {'batch_size': 256,
              'shuffle': False,
              'num_workers': 1}

    prefix = 'tiny_taxinet'
    train_test_split_list = ['train', 'validation', 'test']

    condition_list = ['afternoon']

    for train_test_split in train_test_split_list:

        tensor_dataset, tensor_dataloader = tiny_taxinet_prepare_dataloader(DATA_DIR, condition_list, train_test_split, params, prefix='downsampled', print_mode = True)


        for batch_idx, data in enumerate(tensor_dataloader):
            x_batch = data[0]
            y_batch = data[1]
            
            print(' ')
            print(' Testing Dataloader ')
            print('x_batch: ', x_batch.shape)
            print('y_batch: ', y_batch.shape)
            print(' ')




