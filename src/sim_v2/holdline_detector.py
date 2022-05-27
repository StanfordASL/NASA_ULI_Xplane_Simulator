import numpy as np
import os
import torch
from hold_network import *

class HoldLineDetector:
    def __init__(self, saved_weights, img_channels=3, img_height=32, img_width=64):
        self.channels = img_channels
        self.height = img_height
        self.width = img_width

        # Instantiate holdline network
        model = HoldlineNetwork()
        model.load_state_dict(torch.load(saved_weights))
        model.eval()
        self.model = model
        
    def get_distance(self, img):
        X = torch.zeros(1, self.channels, self.height, self.width)
        X[0,:,:,:] = torch.tensor(img).reshape((self.channels, self.height, self.width)).float()
        mu, sigma = self.model(X)
        y = mu.detach().numpy()[0]

        return y 