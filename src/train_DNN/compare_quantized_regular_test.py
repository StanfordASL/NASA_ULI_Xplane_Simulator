"""
    Tests a pre-trained DNN model
"""

import time
import copy
import torch
import numpy as np
from tqdm.autonotebook import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter

from model_taxinet import TaxiNetDNN, freeze_model, QuantTaxiNetDNN
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

        if print_mode:
            print('inputs: ', inputs.shape)
            print('outputs: ', outputs[0], outputs.shape)
            print('labels: ', labels[0], labels.shape)

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


    # where the training results should go
    results_dir = remove_and_create_dir(SCRATCH_DIR + '/test_DNN_taxinet/')

    # tests the pre-trained model
    model_dir = NASA_ULI_ROOT_DIR + '/pretrained_DNN/' 
    #'epoch_30_resnet18/'
    
    quantized_model_dir = NASA_ULI_ROOT_DIR + '/pretrained_DNN/epoch20_quantized_resnet18/'

    condition_list = ['afternoon', 'morning', 'night', 'overcast']
    #condition_list = ['afternoon']

    # larger images require a resnet, downsampled can have a small custom DNN
    dataset_type = 'large_images'

    train_test_val = 'test'

    with open(results_dir + '/results.txt', 'w') as f:
        header = '\t'.join(['condition', 'train_test_val', 'dataset_type', 'model_type', 'loss', 'mean_inference_time_sec', 'std_inference_time_sec']) 
        f.write(header + '\n')

        for condition in condition_list:
            # where raw images and csvs are saved
            BASE_DATALOADER_DIR = DATA_DIR + '/' + dataset_type  + '/' + condition

            test_dir = BASE_DATALOADER_DIR + '/' + condition + '_' + train_test_val

            dataloader_params = {'batch_size': 128,
                                 'shuffle': False,
                                 'num_workers': 12,
                                 'drop_last': False}

            torch.cuda.empty_cache()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            print('found device: ', device)
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

            out_str = '\t'.join([condition, train_test_val, dataset_type, test_results['model_type'], str(test_results['losses']), str(test_results['time_per_item']), str(test_results['time_per_item_std'])]) 
            f.write(out_str + '\n')

            # COMPARE WITH A RANDOM, UNTRAINED DNN
            untrained_model = TaxiNetDNN()
            untrained_model = untrained_model.to(device)

            test_results = test_model(untrained_model, test_dataset, test_loader, device, loss_func)
            test_results['model_type'] = 'untrained'

            out_str = '\t'.join([condition, train_test_val, dataset_type, test_results['model_type'], str(test_results['losses']), str(test_results['time_per_item']), str(test_results['time_per_item_std'])]) 
            f.write(out_str + '\n')


            # COMPARE WITH A QUANTIZED TRAINED RESNET-18
            device = torch.device("cpu")
            # MODEL
            # instantiate the model and freeze all but penultimate layers
            quantized_model = QuantTaxiNetDNN()

            quantized_model = quantized_model.to(device)
            quantized_model.load_state_dict(torch.load(quantized_model_dir + '/best_model.pt', map_location=torch.device('cpu')))

            test_results = test_model(quantized_model, test_dataset, test_loader, device, loss_func)
            test_results['model_type'] = 'quantized'

            out_str = '\t'.join([condition, train_test_val, dataset_type, test_results['model_type'], str(test_results['losses']), str(test_results['time_per_item']), str(test_results['time_per_item_std'])]) 
            f.write(out_str + '\n')
