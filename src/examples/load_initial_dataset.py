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
    NUM_PRINT = 2

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

    # where the labels for each image (such as distance to centerline) are present
    label_file = data_dir + '/labels.csv'
 
    # dataframe of labels
    labels_df = pandas.read_csv(label_file, sep=',')
    
    # columns are: 
    # ['image_filename', 'absolute_time_GMT_seconds', 'relative_time_seconds', 'distance_to_centerline_meters', 'distance_to_centerline_NORMALIZED', 'downtrack_position_meters', 'downtrack_position_NORMALIZED', 'heading_error_degrees', 'heading_error_NORMALIZED', 'period_of_day', 'cloud_type']

    # loop through images and save in a dataloader
    image_tensor_list = []

    # tensor of targets y:  modify to whatever you want to predict from DNN
    target_tensor_list = []

    for i, image_name in enumerate(image_list):

        # open images and apply transforms
        fname = data_dir + '/' + str(image_name)
        image = Image.open(fname).convert('RGB')
        tensor_image_example = tfms(image)

        # add image
        image_tensor_list.append(tensor_image_example)

        # get the corresponding state information (labels) for each image
        specific_row = labels_df[labels_df['image_filename'] == image_name]
        # there are many states of interest, you can modify to access which ones you want
        dist_centerline_norm = specific_row['distance_to_centerline_NORMALIZED'].item()
        # normalized downtrack position
        downtrack_position_norm = specific_row['downtrack_position_NORMALIZED'].item()

        # normalized heading error
        heading_error_norm = specific_row['heading_error_NORMALIZED'].item()

        # add tensor
        target_tensor_list.append([dist_centerline_norm, downtrack_position_norm, heading_error_norm])

        # periodically save the images to disk 
        if i % NUM_PRINT == 0:
            plt.imshow(image)
            # original image
            title_str = ' '.join(['Dist Centerline: ', str(round(dist_centerline_norm,3)), 'Downtrack Pos. Norm: ', str(round(downtrack_position_norm,3)), '\n', 'Heading Error Norm: ', str(round(heading_error_norm, 3))]) 
            plt.title(title_str)
            plt.savefig(visualization_dir + '/' + str(i) + '.png')
            plt.close()

            # resized and normalized image that can be passed to a DNN
            torchvision.utils.save_image(tensor_image_example, visualization_dir + '/resized_transform_' + str(image_name))
     
        # early terminate for debugging
        if i > MAX_FILES:
            break

    # first, save image tensors
    # concatenate all image tensors
    all_image_tensor = torch.stack(image_tensor_list)
    print(all_image_tensor.shape)

    # save tensors to disk 
    image_data = DATALOADER_DIR + '/images_xplane.pt'
    # sizes are: 126 images, 3 channels, 224 x 224 each 
    # torch.Size([126, 3, 224, 224])
    torch.save(all_image_tensor, image_data)

    ###################################
    # second, save target label tensors
    target_tensor = torch.tensor(target_tensor_list)
    print(target_tensor.shape)
    
    # size: 126 numbers by 3 targets 
    # torch.Size([126])

    # save tensors to disk 
    target_data = DATALOADER_DIR + '/targets_xplane.pt'
    torch.save(target_tensor, target_data)
