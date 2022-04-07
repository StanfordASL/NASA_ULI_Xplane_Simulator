import numpy as np
import julia
from julia import Main

class HoldLineDetector:
    def __init__(self):
        None # TODO
        
    def get_distance(self, img):
        img2 = np.zeros((32, 64, 3, 1))
        img2[:,:,:,0] = img
        Main.x = img2
        out = Main.eval("forward(m, x)")
        return out[0] 