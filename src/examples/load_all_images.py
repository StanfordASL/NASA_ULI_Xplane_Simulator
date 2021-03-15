import sys, os
import torch
import numpy as np
import pandas
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

BASE_DIR = '..'
DATA_DIR = BASE_DIR + '/data/'
VIZ_DIR = BASE_DIR + '/viz/'
DATALOADER_DIR = BASE_DIR + '/loaders/'

NUM_PRINT = 15

if __name__ == '__main__':

    episode_num = 1

    MAX_FILES = np.inf

    # where csvs are 
    episode_dir = DATA_DIR + '/episode_' + str(episode_num)
    
    csv_file =  DATA_DIR + '/' + str(episode_num) + '.csv'

    df = pandas.read_csv(csv_file)

    num_entries = df.shape[0]

    # prepare image transforms
    tfms = transforms.Compose([transforms.Resize((224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]),])
    tensor_list = []
    for i in range(num_entries):
        print(i + 1)
        fname = episode_dir + '/' + str(i+1) + '.png'
        image = Image.open(fname).convert('RGB')
        tensor_image_example = tfms(image)
        print(tensor_image_example.shape)

        tensor_list.append(tensor_image_example.type(torch.float16))

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
