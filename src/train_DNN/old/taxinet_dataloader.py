import sys, os
import torch
import torchvision
import pandas as pd

from PIL import Image
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

import time

BASE_DIR = '..'
DATA_DIR = BASE_DIR + '/data/train/'
VIZ_DIR = BASE_DIR + '/viz/'
DATALOADER_DIR = BASE_DIR + '/loaders/'

class BallCatchDataset(torchvision.datasets.ImageFolder):
    'Characterizes a dataset for PyTorch'
    def __init__(self, episode_ID, past_window_W = 5, future_horizon_H = 10):
        'Initialization'
        # dataset parameters
        self.episode_ID = episode_ID
        self.past_window_W = past_window_W
        self.future_horizon_H = future_horizon_H

        # image data path and time series data loading
        self.image_data_dir = DATA_DIR + 'image/' + 'Episode ' + str(self.episode_ID) + '/'
        time_series_data_dir = DATA_DIR + '/time_series/'
        time_series_df = pd.read_csv(time_series_data_dir + str(self.episode_ID) + '.csv')
        self.time_series_data = torch.tensor(time_series_df.to_numpy())

        # metadata
        self.episode_len = self.time_series_data.shape[0]

        # image transforms
        self.tfms = transforms.Compose([transforms.Resize((224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225]),])

    def __len__(self):
        'Denotes the total number of samples'
        num_valid_timesteps = self.episode_len - (self.future_horizon_H) - (self.past_window_W)

        return num_valid_timesteps

    def __getitem__(self, index):
        'Generates one sample of data'


        # past window W of images
        current_time = index + self.past_window_W
        start_time = current_time - self.past_window_W
        end_time = current_time

        # past window W
        image_tensor_list = []
        for i in range(start_time, end_time):
            start = time.time()
            fname = self.image_data_dir + str(i + 1) + '.png'
            image = Image.open(fname).convert('RGB')
            image_tensor = self.tfms(image)
            image_tensor_list.append(image_tensor)
            end = time.time()

        image_sequence = torch.stack(image_tensor_list)

        print("current_time " + str(current_time) + ": ", end-start)

        # TBD: future controls of MPC x_sol, u_sol
        # TBD: should come from csv
        return image_sequence

if __name__ == '__main__':
    episode_num = 1

    ball_dataset = BallCatchDataset(episode_num)

    params = {'batch_size': 32,
              'shuffle': False,
              'num_workers': 1}

    ball_dataloader = DataLoader(ball_dataset, **params)

    start = time.time()
    for i, image_batch in enumerate(ball_dataloader):
        print(image_batch.shape)
    end = time.time()
    print("Full Data Loader Iteration Time (s): ", (end - start))

