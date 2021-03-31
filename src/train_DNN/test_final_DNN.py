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

DATA_DIR = NASA_ULI_ROOT_DIR + '/data/'

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *

def test_model(model, dataset, dataloader, device, loss_func):
    
    dataset_size = len(dataset)
    
    losses = []
    times_per_item = []
    loss_name = "loss"
    
    # Iterate over data.
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
    
        # forward
        # track history if only in train
        start = time.time()
        outputs = model(inputs)
        end = time.time()

        #print('inputs: ', inputs.shape)
        #print('outputs: ', outputs[0], outputs.shape)
        #print('labels: ', labels[0], labels.shape)

        times_per_item.append( (end - start)/inputs.shape[0] )
        
        loss = loss_func(outputs, labels).mean()
 
        losses.append( loss.cpu().detach().numpy() )

    results = {
        "loss_name": loss_name,
        "losses": np.mean(losses),
        "time_per_item": np.mean(times_per_item),
        "time_per_item_std": np.std(times_per_item) / np.sqrt(dataset_size),
    }
    
    print(results)

    return results


if __name__=='__main__':

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('found device: ', device)

    # where the training results should go
    results_dir = remove_and_create_dir(SCRATCH_DIR + '/test_DNN_taxinet/')

    model_dir = NASA_ULI_ROOT_DIR + '/model/epoch_30_resnet18/'

    # where raw images and csvs are saved
    BASE_DATALOADER_DIR = DATA_DIR + 'nominal_conditions'

    test_dir = BASE_DATALOADER_DIR + '_test/'

    dataloader_params = {'batch_size': 128,
                         'shuffle': False,
                         'num_workers': 12,
                         'drop_last': False}

    # MODEL
    # instantiate the model 
    model = TaxiNetDNN()

    # load the pre-trained model
    if device.type == 'cpu':
        model.load_state_dict(torch.load(model_dir + '/best_model.pt', map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_dir + '/best_model.pt'))
    
    model = model.to(device)
    model.eval()

    # DATALOADERS
    # instantiate the model and freeze all but penultimate layers
    test_dataset = TaxiNetDataset(test_dir)

    test_loader = DataLoader(test_dataset, **dataloader_params)

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

    with open(results_dir + '/results.txt', 'w') as f:
        for k,v in test_results.items():
            out_str = '\t'.join([str(k), str(v)]) + '\n'
            f.write(out_str)


    untrained_model = TaxiNetDNN()
    untrained_model = untrained_model.to(device)

    # COMPARE WITH A RANDOM, UNTRAINED DNN
    test_results = test_model(untrained_model, test_dataset, test_loader, device, loss_func)
    test_results['model_type'] = 'untrained'

    with open(results_dir + '/results.txt', 'a') as f:
        f.write('\n')
        for k,v in test_results.items():
            out_str = '\t'.join([str(k), str(v)]) + '\n'
            f.write(out_str)
