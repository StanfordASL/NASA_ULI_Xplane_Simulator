"""
    Tests a pre-trained DNN model
"""

import time
import copy
import torch
import numpy as np
from tqdm.autonotebook import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter

from model_taxinet import TaxiNetDNN, freeze_model
from taxinet_dataloader import *
from plot_utils import *

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']

DATA_DIR = os.environ['NASA_DATA_DIR'] 

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *

def test_model(model, dataset, dataloader, device, loss_func, print_mode = False):
    
    dataset_size = len(dataset)
    
    perception_losses = []
    control_losses = []
    times_per_item = []
    loss_name = "loss"

    # reference mpc
    target_radius = 1.
    obj_weights = (1.02, 5., 0.3)
    ctrl_lims = [-15., 15.]
    gt_MPC = create_target_landing_cvxpylayer(target_radius, obj_weights, ctrl_lims, n_dim=1, dt=0.1, N=15)


    # Iterate over data.
    for x_vector, p_noisy, p_true, v_robot in tqdm(dataloader):
        
        x_vector = x_vector.to(device)
        p_noisy = p_noisy.to(device)
        p_true = p_true.to(device)
        v_robot = v_robot.to(device)

        # forward
        # track history if only in train
        start = time.time()
        phat, u_mpc, x_mpc = model(x_vector, v_robot)
        end = time.time()

        # get the optimal solution
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

        times_per_item.append( (end - start)/inputs.shape[0] )
        
        perception_losses.append( perception_loss.cpu().detach().numpy() )
        control_losses.append( control_loss.cpu().detach().numpy() )

    results = {
        "loss_name": loss_name,
        "perception_losses": np.mean(perception_losses),
        "control_losses": np.mean(control_losses),
        "time_per_item": np.mean(times_per_item),
        "time_per_item_std": np.std(times_per_item) / np.sqrt(dataset_size),
    }
    
    print(results)

    return results


if __name__=='__main__':

    num_samples = 1000

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('found device: ', device)

    # where the training results should go
    results_dir = remove_and_create_dir(SCRATCH_DIR + '/test_codesign_lander/')

    # larger images require a resnet, downsampled can have a small custom DNN
    dataset_type = 'synthetic_1d'

    train_test_val = 'test'

    # list of all the biases we have
    experiment_list = [0.0, 0.5, 1.0]
    experiment_list = [0.5]
    codesign_mode_list = [True, False]

    dataloader_params = {'batch_size': 32,
                         'shuffle': False,
                         'num_workers': 12,
                         'drop_last': False}


    with open(results_dir + '/results.txt', 'w') as f:
        header = '\t'.join(['bias', 'train_type', 'train_test_val', 'dataset_type', 'model_type', 'perception_loss', 'control_loss', 'mean_inference_time_sec', 'std_inference_time_sec']) 
        f.write(header + '\n')

        for codesign_mode in codesign_mode_list:

            train_type = 'codesign_' + str(codesign_mode)

            for bias in experiment_list:
                bias_str = '_'.join(['bias', str(bias), 'codesign', str(codesign_mode)])

                # where the training results should go
                model_dir = SCRATCH_DIR + '/codesign/' + bias_str + '/'

                # MODEL
                # instantiate the model 
                model = TaskNet()

                # load the pre-trained model
                if device.type == 'cpu':
                    model.load_state_dict(torch.load(model_dir + '/best_model.pt', map_location=torch.device('cpu')))
                else:
                    model.load_state_dict(torch.load(model_dir + '/best_model.pt'))
                
                model = model.to(device)
                model.eval()

                # DATALOADERS
                # instantiate the model and freeze all but penultimate layers
                test_dataset, test_loader = create_synthetic_perception_training_data(num_samples = num_samples, bias = bias, print_mode = False, params = dataloader_params)

                # LOSS FUNCTION
                loss_func = torch.nn.MSELoss().to(device)

                # DATASET INFO
                datasets = {}
                datasets['test'] = test_dataset

                dataloaders = {}
                dataloaders['test'] = test_loader

                # TEST THE TRAINED DNN
                test_results = test_model(model, test_dataset, test_loader, device, loss_func)
                test_results['model_type'] = 'trained'

                out_str = '\t'.join([bias, train_type, train_test_val, dataset_type, test_results['model_type'], str(test_results['losses']), str(test_results['time_per_item']), str(test_results['time_per_item_std'])]) 
                f.write(out_str + '\n')

                # COMPARE WITH A RANDOM, UNTRAINED DNN
                untrained_model = TaxiNetDNN()
                untrained_model = untrained_model.to(device)

                test_results = test_model(untrained_model, test_dataset, test_loader, device, loss_func)
                test_results['model_type'] = 'untrained'

                out_str = '\t'.join([bias, train_type, train_test_val, dataset_type, test_results['model_type'], str(test_results['losses']), str(test_results['time_per_item']), str(test_results['time_per_item_std'])]) 
                f.write(out_str + '\n')
