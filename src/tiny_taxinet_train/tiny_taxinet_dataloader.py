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

def tiny_taxinet_prepare_dataloader(DATA_DIR, condition_list, train_test_split, params, prefix='downsampled', print_mode = True, num_y = 2):

    local_dataset_list = []

    x_train_total = np.array([])
    y_train_total = np.array([])

    for i, condition in enumerate(condition_list):
        # where original XPLANE images are stored 
        label_file = DATA_DIR + 'downsampled/' + condition + '/' + '_'.join([condition, train_test_split, 'downsampled']) + '.h5'


        f = h5py.File(label_file, "r")
        # get the training data, plot a few random images
        x_train = f['X_train'].value.astype(np.float32)
        y_train = f['y_train'].value.astype(np.float32)[:, 0:num_y]

        if print_mode:
            print(' ')
            print('condition: ', condition, ', train_test_split: ', train_test_split)
            print('X shape: ', x_train.shape)
            print('y shape: ', y_train.shape)
            print(' ')

        if i == 0:
            x_train_total = x_train
            y_train_total = y_train
        else:
            #print('x_train_total.shape: ', x_train_total.shape)
            #print('x_train.shape: ', x_train.shape)
            x_train_total = np.vstack((x_train_total, x_train))
            y_train_total = np.vstack((y_train_total, y_train))

        #local_tensor_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        #local_dataset_list.append(local_tensor_dataset)

    #print('x_train_total: ', x_train_total.shape)
    #print('y_train_total: ', y_train_total.shape)

    # tensor dataset
    tensor_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train_total), torch.tensor(y_train_total))
    #tensor_dataset = torch.utils.data.ConcatDataset(local_dataset_list)

    print(' ')
    print('overall dataset size: ', tensor_dataset.__len__())
    print(' ')

    tensor_dataloader = torch.utils.data.DataLoader(tensor_dataset, **params)

    return tensor_dataset, tensor_dataloader


if __name__ == '__main__':

    params = {'batch_size': 256,
              'shuffle': False,
              'num_workers': 1}

    prefix = 'tiny_taxinet'
    train_test_split_list = ['train', 'validation', 'test']

    condition_list = ['afternoon', 'morning', 'overcast', 'night']

    for train_test_split in train_test_split_list:

        tensor_dataset, tensor_dataloader = tiny_taxinet_prepare_dataloader(DATA_DIR, condition_list, train_test_split, params, prefix='downsampled', print_mode = True)

        for batch_idx, data in enumerate(tensor_dataloader):
            x_batch = data[0]
            y_batch = data[1]
            
            #print(' ')
            #print(' Testing Dataloader ')
            #print('x_batch: ', x_batch.shape)
            #print('y_batch: ', y_batch.shape)
            #print(' ')




