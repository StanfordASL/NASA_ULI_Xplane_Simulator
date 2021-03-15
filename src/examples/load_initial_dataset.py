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

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']

from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

DATA_DIR = BASE_DIR + '/data/'
NUM_PRINT = 15

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = BASE_DIR + '/scratch/'

UTILS_DIR = BASE_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *

if __name__ == '__main__':

    # create a temp dir to visualize a few images
    visualization_dir = remove_and_create_dir(SCRATCH_DIR + '/viz/')

    # where the final dataloader will be saved
    DATALOADER_DIR = remove_and_create_dir(SCRATCH_DIR + '/dataloader/')

    MAX_FILES = np.inf

    # where original XPLANE images are stored 
    data_dir = DATA_DIR + '/test_dataset/'
   
    # resize to 224 x 224 x 3 for EfficientNets
    IMAGE_SIZE = 224
    # prepare image transforms
    tfms = transforms.Compose([transforms.Resize((224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]),])

    # loop through images and save in a dataloader
    tensor_list = []
    for i in range(num_entries):

        # open images and apply transforms
        print(i)
        fname = data_dir + '/' + str(i) + '.png'
        image = Image.open(fname).convert('RGB')
        tensor_image_example = tfms(image)
        print(tensor_image_example.shape)

        tensor_list.append(tensor_image_example)

        # periodically save the images to disk 
        if i % NUM_PRINT == 0:
            plt.imshow(image)
            plt.savefig(visualization_dir + '/' + str(i+1) + '.png')
     
        # early terminate for debugging
        if i > MAX_FILES:
            break

    all_image_tensor = torch.stack(tensor_list)
    print(all_image_tensor.shape)

    image_data = DATALOADER_DIR + '/xplane_images.pt'
    torch.save(all_image_tensor, image_data)
