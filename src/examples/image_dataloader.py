'''
    Goal: Given saved .pt files for images and targets from X-Plane, 
    load a pytorch dataloader that can be used for training a DNN
'''

import sys, os
import torch
import numpy as np
import pandas
import matplotlib.pyplot as plt
import time

from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader


# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']

DATA_DIR = NASA_ULI_ROOT_DIR + '/data/'

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *

class TaxiNetDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dataloader_dir, num_expected_targets=2):

        'Initialization'
        self.dataloader_dir = dataloader_dir
        
        # Load data and get label
        self.image_data = torch.load(self.dataloader_dir + '/images_xplane.pt')
        self.target_data = torch.load(self.dataloader_dir + '/targets_xplane.pt')

  def __len__(self):
       
        # number of images in dataset
        num_images = self.image_data.shape[0]
        return num_images

  def __getitem__(self, index):
        'Generates one sample of data'

        all_image_tensor = self.image_data[index]
        target_tensor = self.target_data[index]

        return all_image_tensor, target_tensor

if __name__ == '__main__':

    DATALOADER_DIR = SCRATCH_DIR + '/dataloader/'

    taxinet_dataset = TaxiNetDataset(DATALOADER_DIR)

    params = {'batch_size': 32,
              'shuffle': False,
              'num_workers': 1}

    taxinet_dataloader = DataLoader(taxinet_dataset, **params)

    for i, (image_batch, target_tensor) in enumerate(taxinet_dataloader):
        print(' ')
        print('image batch: ', image_batch.shape)
        print('target tensor: ', target_tensor.shape)
        print(' ')
 
