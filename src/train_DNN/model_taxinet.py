import torch
import torch.nn as nn
from torchvision import models
#import torchvision.models.quantization as quant_models

'''
EfficientNet model as CNN feature extractor
Takes in images and outputs embeddings
Input:
(N, C=3, H=224, W=224) = (batch size, channels, height, width)
Output:
(N, E=1280) = (batch size, feature number)
'''

class TaxiNetDNN(nn.Module):
    def __init__(self, model_name="resnet18", quantize=False, y_dim=3):
        super(TaxiNetDNN, self).__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=True)
        elif model_name == 'squeezenet':
            self.model = models.squeezenet1_1(pretrained=True)

        if quantize:
            self.model = quant_models.resnet18(pretrained=True, quantize=True)

        self.model.fc = nn.Linear(self.model.fc.in_features, y_dim)
        self.fc = self.model.fc

    def forward(self, z):
        out = self.model(z)
        return out

def QuantTaxiNetDNN(y_dim=3):
    # You will need the number of filters in the `fc` for future use.
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_fe = quant_models.resnet18(pretrained=True, progress=True, quantize=True)
    num_ftrs = model_fe.fc.in_features

    # Step 1. Isolate the feature extractor.
    model_fe_features = nn.Sequential(
    model_fe.quant,  # Quantize the input
    model_fe.conv1,
    model_fe.bn1,
    model_fe.relu,
    model_fe.maxpool,
    model_fe.layer1,
    model_fe.layer2,
    model_fe.layer3,
    model_fe.layer4,
    model_fe.avgpool,
    model_fe.dequant,  # Dequantize the output
    )

    # Step 2. Create a new "head"
    new_head = nn.Sequential(
    nn.Linear(num_ftrs, y_dim),
    )

    # Step 3. Combine, and don't forget the quant stubs.
    new_model = nn.Sequential(
    model_fe_features,
    nn.Flatten(1),
    new_head,
    )
    
    return new_model



def freeze_model(model, freeze_frac=True):
    # freeze everything
    n_params = len(list(model.parameters()))
    for i, p in enumerate(model.parameters()):
        #if i < 6*n_params/7:
        if i < 4*n_params/7:
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


