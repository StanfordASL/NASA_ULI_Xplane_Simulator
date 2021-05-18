import torch
import torch.nn as nn
from torchvision import models

'''
    small model for tiny taxinet
'''

class TinyTaxiNetDNN(nn.Module):
    def __init__(self, model_name="TinyTaxiNet"):

        super(TinyTaxiNetDNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        
        y_dim = 2
        
        self.model.fc = nn.Linear(self.model.fc.in_features, y_dim)
        
        self.fc = self.model.fc

    def forward(self, z):
        out = self.model(z)
        return out

