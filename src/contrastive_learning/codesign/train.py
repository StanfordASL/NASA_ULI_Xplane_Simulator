"""
"""

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

if __name__=='__main__':

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_options = {"epochs": 10,
                     "learning_rate": 1e-3, 
                     }

    dataloader_params = {'batch_size': 32,
                         'shuffle': True,
                         'num_workers': 1,
                         'drop_last': False,
                         'pin_memory': True}

    print('found device: ', device)

    # list of all the biases we have
    experiment_list = [[0.0], [0.5], [1.0]]

    for condition_list in experiment_list:
    
        condition_str = '_'.join(condition_list)

        # where the training results should go
        results_dir = remove_and_create_dir(SCRATCH_DIR + '/codesign/' + condition_str + '/')

        # MODEL
        # instantiate the model and freeze all but penultimate layers
        model = DNN()

        # DATALOADERS
        # instantiate the model and freeze all but penultimate layers
        train_dataset, train_loader = tiny_taxinet_prepare_dataloader(DATA_DIR, condition_list, 'train', dataloader_params)

        val_dataset, val_loader = tiny_taxinet_prepare_dataloader(DATA_DIR, condition_list, 'validation', dataloader_params)


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

