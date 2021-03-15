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


if __name__ == '__main__':

    MAX_FILES = np.inf

    # where csvs are 
    data_dir = DATA_DIR + '/test_dataset/'
   
    # resize to 224 x 224 x 3 for EfficientNets
    IMAGE_SIZE = 224
    # prepare image transforms
    tfms = transforms.Compose([transforms.Resize((224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]),])

    tensor_list = []
    for i in range(num_entries):
        print(i)
        fname = data_dir + '/' + str(i) + '.png'
        image = Image.open(fname).convert('RGB')
        tensor_image_example = tfms(image)
        print(tensor_image_example.shape)

        tensor_list.append(tensor_image_example)

        if i % NUM_PRINT == 0:
            plt.imshow(image)
            plt.savefig(VIZ_DIR + '/' + str(i+1) + '.png')
      
        if i > MAX_FILES:
            break

    all_image_tensor = torch.stack(tensor_list)
    print(all_image_tensor.shape)

    timeseries = torch.tensor(df.values).type(torch.float16)

    episode_data = DATALOADER_DIR + '/images_episode_' + str(episode_num) + '.pt'
    timeseries_data = DATALOADER_DIR + '/timeseries_episode_' + str(episode_num) + '.pt'

    torch.save(all_image_tensor, episode_data)
    torch.save(timeseries, timeseries_data) 

