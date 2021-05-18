import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

'''
    small model for tiny taxinet

    Follows the SISL Kochenderfer Lab architecture:

    Number of layers: 4
    Number of inputs: 128
    Number of outputs: 2
    Maximum layer size: 128

    2nd line: 128,16,8,8,2

    Input size: 128
    Layer 2: 16
    Layer 3: 8
    Layer 4: 8
    Output: 2

'''

class TinyTaxiNetDNN(nn.Module):
    def __init__(self, model_name="TinyTaxiNet"):
        super(TinyTaxiNetDNN, self).__init__()

        self.fc1 = torch.nn.Linear(128, 16)
        self.fc2 = torch.nn.Linear(16, 8)
        self.fc3 = torch.nn.Linear(8, 8)
        self.fc4 = torch.nn.Linear(8, 2)

    def forward(self, z):
        x = torch.flatten(z, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # remember no relu on last layer!
        x = self.fc4(x)

        return x



