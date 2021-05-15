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

from textfile_utils import *

def show(img, fname_path, title=''):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), cmap='gray')
    plt.title(title)
    plt.savefig(fname_path)

if __name__ == '__main__':
    # how often to plot a few images for progress report
    # warning: plotting is slow
    NUM_PRINT = 5
    prefix = 'tiny_taxinet'
    train_test_split_list = ['train', 'validation', 'test']

    MAX_IMAGES = 30
    NROW = 5

    for condition in ['afternoon', 'morning', 'night', 'overcast']:
        # create a temp dir to visualize a few images
        visualization_dir = SCRATCH_DIR + '/' + prefix + '/' + condition
        remove_and_create_dir(visualization_dir)
        for train_test_split in train_test_split_list:

            # where original XPLANE images are stored 
            label_file = DATA_DIR + 'downsampled/' + condition + '/' + '_'.join([condition, train_test_split, 'downsampled']) + '.h5'

            print('label_file: ', label_file)

            f = h5py.File(label_file, "r")
            # get the training data, plot a few random images
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
            image_array_np = x_train[start_index: start_index + MAX_IMAGES]
            tensor_array = torch.tensor(image_array_np).unsqueeze(1)
            
            # grid array
            ####################
            grid = torchvision.utils.make_grid(tensor_array, nrow=NROW)
            fname_path = visualization_dir + '/' + condition + '_' + train_test_split + '.png'
            show(grid, fname_path, title=condition)
            f.close()

            # now plot images and corresponding state information


