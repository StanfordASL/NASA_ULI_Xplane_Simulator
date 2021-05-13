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
    condition = 'afternoon'
    train_test_split = 'train'

    MAX_IMAGES = 30
    NROW = 5

    for condition in ['afternoon', 'morning', 'night', 'overcast']:

        # create a temp dir to visualize a few images
        visualization_dir = SCRATCH_DIR + '/' + prefix + '/' + condition
        remove_and_create_dir(visualization_dir)

        # where original XPLANE images are stored 
        label_file = DATA_DIR + 'downsampled/' + condition + '/' + '_'.join([condition, train_test_split, 'downsampled']) + '.h5'

        print('label_file: ', label_file)

        f = h5py.File(label_file, "r")
        # List all groups
        print("Keys: %s" % f.keys())
        for i in range(len(f.keys())):
            a_group_key = list(f.keys())[i]
            print('key: ', a_group_key)

        # get the training data, plot a few random images
        x_train = f['X_train'].value
        
        ## plot images 1 by 1
        #for image_index in range(MAX_IMAGES):
        #    image_numpy = x_train[image_index]
        #    tensor_image = torch.tensor(image_numpy)
        #    fname = visualization_dir + '/' + condition + '_' + str(image_index) + '.png'
        #    torchvision.utils.save_image(tensor_image, fname)

        # plot vector of images
        ####################
        num_images_total = x_train.shape[0]

        start_index = np.random.randint(num_images_total - MAX_IMAGES)
        print('start_index: ', start_index)

        image_array_np = x_train[start_index: start_index + MAX_IMAGES]

        tensor_array = torch.tensor(image_array_np).unsqueeze(1)
        print(' ')
        print('tensor_array shape: ', tensor_array.shape)
        print(' ')
        
        # grid array
        grid = torchvision.utils.make_grid(tensor_array, nrow=NROW)
        fname_path = visualization_dir + '/' + condition + '.png'
        show(grid, fname_path, title=condition)

        f.close()
