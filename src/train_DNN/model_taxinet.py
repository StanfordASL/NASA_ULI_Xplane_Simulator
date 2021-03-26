import torch
import torch.nn as nn
from torchvision import models

'''
EfficientNet model as CNN feature extractor
Takes in images and outputs embeddings
Input:
(N, C=3, H=224, W=224) = (batch size, channels, height, width)
Output:
(N, E=1280) = (batch size, feature number)
'''

class TaxiNetDNN(nn.Module):
    def __init__(self, model_name="efficientnet-b0"):
        super(TaxiNetDNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        y_dim = 2
        self.model.fc = nn.Linear(self.model.fc.in_features, y_dim)
        self.fc = self.model.fc

    def forward(self, z):
        out = self.model(z)
        return out

def freeze_model(model, freeze_frac=True):
    # freeze everything
    n_params = len(list(model.parameters()))
    for i, p in enumerate(model.parameters()):
        if i < 6*n_params/7:
            p.requires_grad = False

    # make last layer trainable
    for p in model.fc.parameters():
        p.requires_grad = True
       
    return model


def unfreeze_model(model):
    global og_req_grads
    # unfreeze everything
    for p,v in zip( model.parameters(), og_req_grads):
        p.requires_grad = v
