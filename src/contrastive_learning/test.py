import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

def show(img, fname_path):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest', cmap='gray')
    plt.savefig(fname_path)

w = torch.randn(10,3,640,640)
w = torch.randn(10,1,8,16)
grid = torchvision.utils.make_grid(w, nrow=2)
fname_path = 'b.png'
show(grid, fname_path)

