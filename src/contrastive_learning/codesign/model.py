import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

'''
    small model for tiny taxinet

'''

class DNN(nn.Module):
    def __init__(self, model_name="perception"):
        super(DNN, self).__init__()

        self.fc1 = torch.nn.Linear(2, 8)
        self.fc2 = torch.nn.Linear(8, 4)
        self.fc3 = torch.nn.Linear(4, 1)

    def forward(self, z):
        x = torch.flatten(z, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # remember no relu on last layer!
        x = self.fc3(x)

        return x


