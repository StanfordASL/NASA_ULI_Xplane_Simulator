"""
"""

import sys, os
import time
import copy
import torch
import numpy as np
from tqdm.autonotebook import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

TRAIN_DNN_DIR = NASA_ULI_ROOT_DIR + '/src/train_DNN/'
sys.path.append(TRAIN_DNN_DIR)

from textfile_utils import *
from plot_utils import *

from model import *
from generate_synthetic_data import create_synthetic_perception_training_data


def train_model(model, datasets, dataloaders, dist_fam, optimizer, device, results_dir, perception_mode = False, num_epochs=25, log_every=100):
    """
    Trains a model on datatsets['train'] using criterion(model(x_vector), p_noisy) as the loss.
    Returns the model with lowest loss on datasets['val']
    Puts model and x_vector on device.
    Trains for num_epochs passes through both datasets.

    Writes tensorboard info to ./runs/ if given
    """
    writer = None
    writer = SummaryWriter(log_dir=results_dir)

    model = model.to(device)

    since = time.time()

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    tr_loss = np.nan
    val_loss = np.nan

    n_tr_batches_seen = 0

    train_loss_vec = []
    val_loss_vec = []

    # reference mpc
    target_radius = 1.
    obj_weights = (1.02, 5., 0.3)
    ctrl_lims = [-15., 15.]
    gt_MPC = create_target_landing_cvxpylayer(target_radius, obj_weights, ctrl_lims, n_dim=1, dt=0.1, N=15)

    with tqdm(total=num_epochs, position=0) as pbar:
        pbar2 = tqdm(total=dataset_sizes['train'], position=1)
        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                running_tr_loss = 0.0 # used for logging

                running_n = 0

                # Iterate over data.
                pbar2.refresh()
                pbar2.reset(total=dataset_sizes[phase])
                for x_vector, p_noisy, p_true, v_robot in dataloaders[phase]:
                    if phase == 'train':
                        n_tr_batches_seen += 1

                    x_vector = x_vector.to(device)
                    p_noisy = p_noisy.to(device)
                    p_true = p_true.to(device)
                    v_robot = v_robot.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # first, run TaskNet and get the predicted error and controls
                        phat, u_mpc, x_mpc = model(x_vector, v_robot)

                        # get the optimal controls (for supervision only)
                        with torch.no_grad():
                            x_robot = x_vector[:,0].clone().detach()
                            v_robot = v_robot.clone().detach()
                            x_target_gt = x_vector[:, 1].clone().detach()
                            x0 = torch.vstack([x_robot, v_robot.squeeze()]).T
                            x_target_check = x_robot - p_true.squeeze()
                            x_opt, u_opt = gt_MPC(x0, x_target_gt.unsqueeze(-1))

                        # always compute perception error
                        ####################
                        perception_loss = loss_func(phat, p_noisy)
                        ####################

                        # compute control error as well
                        control_loss = loss_func(u_mpc, u_opt) + loss_func(x_mpc, x_opt)

                        # depending on mode, select which loss to use for optimization
                        if perception_mode:
                            loss = perception_loss
                        else:
                            loss = control_loss

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * x_vector.shape[0]

                    if phase =='train':
                        running_n += x_vector.shape[0]
                        running_tr_loss += loss.item() * x_vector.shape[0]

                        if n_tr_batches_seen % log_every == 0:
                            mean_loss = running_tr_loss / running_n

                            writer.add_scalar('loss/train', mean_loss, n_tr_batches_seen)

                            running_tr_loss = 0.
                            running_n = 0

                    pbar2.set_postfix(split=phase, batch_loss=loss.item())
                    pbar2.update(x_vector.shape[0])



                epoch_loss = running_loss / dataset_sizes[phase]

                if phase == 'train':
                    tr_loss = epoch_loss
                    train_loss_vec.append(tr_loss)

                if phase == 'val':
                    val_loss = epoch_loss
                    writer.add_scalar('loss/val', val_loss, n_tr_batches_seen)
                    val_loss_vec.append(val_loss)

                pbar.set_postfix(tr_loss=tr_loss, val_loss=val_loss)

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            pbar.update(1)

            print(' ')
            print('training loss: ', train_loss_vec)
            print('val loss: ', val_loss_vec)
            print(' ')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    writer.flush()

    # plot the results to a file
    plot_file = results_dir + '/loss.pdf'
    basic_plot_ts(train_loss_vec, val_loss_vec, plot_file, legend = ['Train Loss', 'Val Loss'])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



if __name__=='__main__':
    num_samples = 1000

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_options = {"epochs": 20,
                     "learning_rate": 1e-3,
                     }

    dataloader_params = {'batch_size': 32,
                         'shuffle': True,
                         'num_workers': 1,
                         'drop_last': False,
                         'pin_memory': True}

    print('found device: ', device)

    # list of all the biases we have
    experiment_list = [0.0, 0.5, 1.0]

    for bias in experiment_list:

        condition_str = 'bias-' + str(bias)

        # where the training results should go
        results_dir = remove_and_create_dir(SCRATCH_DIR + '/codesign/' + condition_str + '/')

        # MODEL
        # instantiate the model and freeze all but penultimate layers
        model = TaskNet()

        # DATALOADERS
        # instantiate the model and freeze all but penultimate layers
        train_dataset, train_loader = create_synthetic_perception_training_data(num_samples = num_samples, bias = bias, print_mode = False, params = dataloader_params)

        val_dataset, val_loader = create_synthetic_perception_training_data(num_samples = num_samples, bias = bias, print_mode = False, params = dataloader_params)

        # OPTIMIZER
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=train_options["learning_rate"],
                                     amsgrad=True)

        # LOSS FUNCTION
        loss_func = torch.nn.MSELoss().to(device)

        # DATASET INFO
        datasets = {}
        datasets['train'] = train_dataset
        datasets['val'] = val_dataset

        dataloaders = {}
        dataloaders['train'] = train_loader
        dataloaders['val'] = val_loader

        # train the DNN
        model = train_model(model, datasets, dataloaders, loss_func, optimizer, device, results_dir, num_epochs=train_options['epochs'], log_every=100)

        # save the best model to the directory
        torch.save(model.state_dict(), results_dir + "/best_model.pt")
