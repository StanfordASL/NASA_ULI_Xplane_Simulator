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

class TaskNet(nn.Module):
    def __init__(self, model_name="tasknet"):
        super(TaskNet, self).__init__()

        self.DNN = DNN()
        self.MPC = MPC()

    def forward(self, x_vector, v_robot):
        x_robot = x_vector[:,0]
        x_target = x_vector[:, 1]

        phat = self.DNN(x_vector)
        u_mpc, x_mpc, J_mpc = self.MPC(x_robot, v_robot, phat)

        return phat, u_mpc, x_mpc, J_mpc

