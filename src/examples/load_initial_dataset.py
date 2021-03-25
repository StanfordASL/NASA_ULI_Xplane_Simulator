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

import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

DATA_DIR = NASA_ULI_ROOT_DIR + '/data/'

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *

if __name__ == '__main__':

    # how often to plot a few images for progress report
    # warning: plotting is slow
    NUM_PRINT = 5

    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224

    # create a temp dir to visualize a few images
    visualization_dir = SCRATCH_DIR + '/viz/'
    remove_and_create_dir(visualization_dir)

    # where the final dataloader will be saved
    DATALOADER_DIR = remove_and_create_dir(SCRATCH_DIR + '/dataloader/')

    MAX_FILES = np.inf

    # where original XPLANE images are stored 
    data_dir = DATA_DIR + '/test_dataset_smaller_ims/'
   
    # resize to 224 x 224 x 3 for EfficientNets
    # prepare image transforms
    # warning: you might need to change the normalization values given your dataset's statistics
    tfms = transforms.Compose([transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]),])

    image_list = [x for x in os.listdir(data_dir) if x.endswith('.png')]

    # loop through images and save in a dataloader
    tensor_list = []
    for i, image_name in enumerate(image_list):

        # open images and apply transforms
        print(i, image_name)
        fname = data_dir + '/' + str(image_name)
        image = Image.open(fname).convert('RGB')
        tensor_image_example = tfms(image)
        print(tensor_image_example.shape)

        tensor_list.append(tensor_image_example)

        # periodically save the images to disk 
        if i % NUM_PRINT == 0:
            plt.imshow(image)
            # original image
            plt.savefig(visualization_dir + '/' + str(i) + '.png')
            plt.close()

            # resized and normalized image that can be passed to a DNN
            torchvision.utils.save_image(tensor_image_example, visualization_dir + '/resized_transform_' + str(image_name))
     
        # early terminate for debugging
        if i > MAX_FILES:
            break

    # concatenate all image tensors
    all_image_tensor = torch.stack(tensor_list)
    print(all_image_tensor.shape)

    # save tensors to disk 
    image_data = DATALOADER_DIR + '/xplane_images.pt'
    torch.save(all_image_tensor, image_data)
