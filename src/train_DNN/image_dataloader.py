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

import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

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
  def __init__(self, dataloader_dir, num_expected_targets=2, IMAGE_WIDTH=224, IMAGE_HEIGHT=224):

        'Initialization'
        # image data path and time series data loading
        self.data_dir = dataloader_dir

        self.image_list = [x for x in os.listdir(self.data_dir) if x.endswith('.png')]

        # where the labels for each image (such as distance to centerline) are present
        label_file = self.data_dir + '/labels.csv'
     
        # dataframe of labels
        self.labels_df = pandas.read_csv(label_file, sep=',')

        # image transforms
        self.tfms = transforms.Compose([transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225]),])



        # optional: do not use normalization or resize
        # image transforms
        #self.tfms = transforms.Compose([transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        #                                transforms.ToTensor()])
        #self.tfms = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
        # number of images in dataset
        num_images = len(self.image_list)
        return num_images

  def __getitem__(self, index):
        'Generates one sample of data'

        image_name = self.image_list[index]

        # open images and apply transforms
        fname = self.data_dir + '/' + str(image_name)
        image = Image.open(fname).convert('RGB')
        tensor_image_example = self.tfms(image)

        # get the corresponding state information (labels) for each image
        specific_row = self.labels_df[self.labels_df['image_filename'] == image_name]
        # there are many states of interest, you can modify to access which ones you want
        dist_centerline_norm = specific_row['distance_to_centerline_NORMALIZED'].item()
        # normalized downtrack position
        downtrack_position_norm = specific_row['downtrack_position_NORMALIZED'].item()

        # add tensor
        target_tensor_list = [dist_centerline_norm, downtrack_position_norm]

        # concatenate all image tensors
        target_tensor = torch.tensor(target_tensor_list)
    
        return tensor_image_example, target_tensor

if __name__ == '__main__':

    print_mode = False

    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224

    visualization_dir = NASA_ULI_ROOT_DIR + '/scratch/test_dataloader_viz/'
    remove_and_create_dir(visualization_dir)

    # where original XPLANE images are stored 
    DATALOADER_DIR = DATA_DIR + '/test_dataset_smaller_ims/'
    DATALOADER_DIR = DATA_DIR + '/medium_size_dataset/nominal_conditions_subset/'

    taxinet_dataset = TaxiNetDataset(DATALOADER_DIR)

    params = {'batch_size': 32,
              'shuffle': False,
              'num_workers': 1}

    taxinet_dataloader = DataLoader(taxinet_dataset, **params)

    NUM_PRINT = 5

    for i, (image_batch, target_tensor) in enumerate(taxinet_dataloader):
        print(' ')
        print('image batch: ', image_batch.shape)
        print('target tensor: ', target_tensor.shape)
        print(' ')

        if (i % NUM_PRINT == 0) and print_mode:
            torchvision.utils.save_image(image_batch, visualization_dir + '/resized_transform_' + str(i) + '.png')

