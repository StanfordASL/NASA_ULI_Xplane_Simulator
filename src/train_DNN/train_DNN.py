import torch
import numpy as np
import sys, os

from model_taxinet import TaxiNetDNN, freeze_model
from taxinet_dataloader import *

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

def train_model(train_options, dataloader_params, val_dataloader_params):

    results_dir = train_options['results_dir']
    train_dataset = TaxiNetDataset(train_options['data_dir'])

    train_loader = DataLoader(train_dataset, **dataloader_params)

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


        #print("Computing Validation Loss.")
        #val_total_loss = 0
        #val_batches = 0
        #model.eval()
        #with torch.no_grad():
        #    for batch_num, (img_seq, x0, x_opt, u_opt, ball_future) in enumerate(val_loader):
        #        try:
        #            val_batches += 1
        #            img_seq, x0, x_opt, u_opt, ball_future = img_seq.to(device), x0.to(device), x_opt.to(device), u_opt.to(device), ball_future.to(device)

        #            x_opt, u_opt, _, _ = gt_controller(x0.unsqueeze(-1),
        #                                               ball_future.float())

        #            # forward pass
        #            img_seq = img_seq.permute(1, 0, 2, 3, 4)
        #            x0 = x0.unsqueeze(-1).type(torch.float32)
        #            x_sol, u_sol = model(img_seq, x0)
        #            # ball_pred = model(img_seq)

        #            # evaluate loss
        #            loss = loss_func(u_sol, u_opt.squeeze())
        #            # loss = loss_func(ball_pred, ball_future.type(torch.float32).squeeze())
        #            val_total_loss += loss.item()
        #        except:
        #            print("Skipping validation batch due to an issue.")
        #model.train()

        #torch.save(model, results_dir + "/Epoch_"+str(i+1))

        print("normalized train loss: ", total_loss / num_batches)
        #print("normalized val loss: ", val_total_loss / val_batches)
        train_loss.append(total_loss / num_batches)
        #val_loss.append(val_total_loss / val_batches)
        np.savetxt(results_dir + '/train_loss.txt', train_loss)
        #np.savetxt('../task_loss/val_loss.txt', val_loss)

    return train_loss

if __name__=='__main__':

    # where the training results should go
    results_dir = remove_and_create_dir(SCRATCH_DIR + '/DNN_train_taxinet/')

    # where raw images and csvs are saved
    DATALOADER_DIR = DATA_DIR + '/test_dataset_smaller_ims/'

    #DATALOADER_DIR = DATA_DIR + '/medium_size_dataset/nominal_conditions_subset/'

    train_options = {"epochs": 20,
                     "learning_rate": 1e-3, 
                     "results_dir": results_dir,
                     "data_dir": DATALOADER_DIR
                     }

    dataloader_params = {'batch_size': 32,
                         'shuffle': True,
                         'num_workers': 12,
                         'drop_last': False}

    val_dataloader_params = {'batch_size': 32,
                             'shuffle': False,
                             'num_workers': 12,
                             'drop_last': False}

    train_loss = train_model(train_options, dataloader_params, val_dataloader_params)

