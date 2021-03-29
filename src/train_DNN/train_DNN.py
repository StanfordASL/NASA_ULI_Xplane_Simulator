import torch
import numpy as np
import sys, os

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

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(train_options, dataloader_params):

    results_dir = train_options['results_dir']
    train_dataset = TaxiNetDataset(train_options['train_dir'])
    val_dataset = TaxiNetDataset(train_options['val_dir'])

    train_loader = DataLoader(train_dataset, **dataloader_params)
    val_loader = DataLoader(val_dataset, **dataloader_params)

    model = TaxiNetDNN()
    model = freeze_model(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train_options["learning_rate"],
                                 amsgrad=True)
    loss_func = torch.nn.MSELoss().to(device)

    train_loss = []
    val_loss = []
    for i in range(train_options['epochs']):
        print("Epoch: ", i + 1)
        total_loss = 0
        num_batches = 0
        for batch_num, (img_seq, targets) in enumerate(train_loader):

            # try:
            num_batches += 1
            img_seq, targets = img_seq.to(device), targets.to(device)

            predictions = model(img_seq)

            if batch_num % 100 == 0 and batch_num > 0:
                print("batch_num: ", batch_num, "total loss: ", total_loss / num_batches)

            # evaluate loss
            loss = loss_func(targets, predictions)
            total_loss += loss.item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print("Computing Validation Loss.")
        val_total_loss = 0
        val_batches = 0
        model.eval()
        with torch.no_grad():
            for batch_num, (img_seq, targets) in enumerate(val_loader):
                val_batches += 1
                img_seq, targets = img_seq.to(device), targets.to(device)

                predictions = model(img_seq)

                # evaluate loss
                loss = loss_func(targets, predictions)
                val_total_loss += loss.item()
        model.train()

        #torch.save(model, results_dir + "/Epoch_"+str(i+1))

        print("normalized train loss: ", total_loss / num_batches)
        print("normalized val loss: ", val_total_loss / val_batches)
        train_loss.append(total_loss / num_batches)
        val_loss.append(val_total_loss / val_batches)
        np.savetxt(results_dir + '/train_loss.txt', train_loss)
        np.savetxt(results_dir + '/val_loss.txt', val_loss)

    plot_file = results_dir + '/loss.pdf' 
    basic_plot_ts(train_loss, val_loss, plot_file, legend = ['Train Loss', 'Val Loss'])

    return train_loss

if __name__=='__main__':
    print('found device: ', device)

    # where the training results should go
    results_dir = remove_and_create_dir(SCRATCH_DIR + '/DNN_train_taxinet/')

    # where raw images and csvs are saved
    BASE_DATALOADER_DIR = DATA_DIR + 'nominal_conditions_'

    train_dir = BASE_DATALOADER_DIR + 'train/'
    val_dir = BASE_DATALOADER_DIR + 'val/'

    train_options = {"epochs": 50,
                     "learning_rate": 1e-3, 
                     "results_dir": results_dir,
                     "train_dir": train_dir, 
                     "val_dir": val_dir
                     }

    dataloader_params = {'batch_size': 512,
                         'shuffle': True,
                         'num_workers': 12,
                         'drop_last': False}

    train_loss = train_model(train_options, dataloader_params)

