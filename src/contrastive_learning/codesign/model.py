import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

'''
    small model for tiny taxinet

'''

def nd_double_integrator_dynamics(dt, n):
    """Returns discrete double integrator dynamics matrices"""
    A = np.eye(2 * n) + dt * np.eye(2 * n, k=n)
    B = (dt**2 / 2 * np.eye(2 * n, n)  + dt * np.eye(2 * n, n, k=-n))

    return A, B

def create_target_landing_cvxpylayer(target_radius, obj_weights, ctrl_lims, n_dim=1, dt=0.1, N=15):
    # decision variables
    x = cp.Variable((2 * n_dim, N)) # states
    u = cp.Variable((n_dim, N)) # controls

    # parameters
    x0 = cp.Parameter(2 * n_dim) # initial state
    x_target = cp.Parameter(n_dim)

    # dynamics constraints
    A, B = nd_double_integrator_dynamics(dt, n=n_dim)
    initial_condition = [x[:, 0] == A @ x0 + B @ u[:, 0]]
    dynamics_constraints = [x[:, i+1] == A @ x[:, i] + B @ u[:, i] for i in range(N-1)]

    # control constraints
    control_constraints = [u >= ctrl_lims[0], u <= ctrl_lims[1]]

    # gather constraints
    constraints = initial_condition + dynamics_constraints + control_constraints

    # objective
    acc_pen, tar_pen, ctrl_pen = obj_weights

    fin_pos = x[:n_dim, -1] # final state position

    acc_term = acc_pen * cp.norm(fin_pos - x_target) # target accuracy

    tar_term = tar_pen * cp.norm(cp.maximum(0, x_target - x[0, :] - target_radius)) # target set membership
    ctrl_term = ctrl_pen * cp.norm(u, 'fro') # control usage

    objective = acc_term + tar_term + ctrl_term

    # create cvxpy problem
    problem =  cp.Problem(cp.Minimize(objective), constraints)
    assert problem.is_dpp()

    # get cvxpylayer
    cvxpylayer = CvxpyLayer(problem, parameters=[x0, x_target], variables=[x, u])

    return cvxpylayer

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

        target_radius = 1.
        obj_weights = (1.02, 5., 0.3)
        ctrl_lims = [-15., 15.]
        self.MPC = create_target_landing_cvxpylayer(target_radius, obj_weights, ctrl_lims, n_dim=1, dt=0.1, N=15)

    def forward(self, x_vector, v_robot):
        x_robot = x_vector[:,0]
        x_target = x_vector[:, 1]

        x0 = torch.vstack([x_robot, v_robot.squeeze()]).T
        phat = self.DNN(x_vector)

        x_target_estimate = x_robot - phat.squeeze()

        x_mpc, u_mpc = self.MPC(x0, x_target_estimate.unsqueeze(-1))

        return phat, u_mpc, x_mpc
