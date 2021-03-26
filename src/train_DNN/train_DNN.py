import torch
import numpy as np

from models import TaskRelevantTransformer
from controllers import SimpleQP
from ball_catch_dataset import *

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device =   'cpu'

def train_model(train_options, dataloader_params, val_dataloader_params):
    controller_settings = {"x_lims": [50, 622],
                           "u_lims": [-400, 400],
                           "dt": 0.02,
                           "N": 7,
                           "control_penalty": 0.025}
    gt_controller = SimpleQP(controller_settings, device)

    train_dataset = create_ball_catch_dataset('../data/train/')
    train_loader = DataLoader(train_dataset, **dataloader_params)

    val_dataset = create_ball_catch_dataset('../data/validation/')
    val_loader = DataLoader(val_dataset, **val_dataloader_params)

    model = TaskRelevantTransformer()
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
        for batch_num, (img_seq, x0, x_opt, u_opt, ball_future) in enumerate(train_loader):

            # try:
            num_batches += 1
            img_seq, x0, x_opt, u_opt, ball_future = img_seq.to(device), x0.to(device), x_opt.to(device), u_opt.to(device), ball_future.to(device)

            x_opt, u_opt, _, _ = gt_controller(x0.unsqueeze(-1),
                                               ball_future.float())

            # forward pass
            img_seq = img_seq.permute(1, 0, 2, 3, 4)
            x0 = x0.unsqueeze(-1).type(torch.float32)
            x_sol, u_sol = model(img_seq, x0)
            # ball_pred = model(img_seq)

            if batch_num % 100 == 0 and batch_num > 0:
                print("batch_num: ", batch_num, "total loss: ", total_loss / num_batches)
                print("u_sol: ", u_sol[:4])
                print("u_opt: ", u_opt[:4])

            # evaluate loss
            loss = loss_func(u_sol, u_opt.squeeze())
            # loss = loss_func(ball_pred, ball_future.type(torch.float32).squeeze())
            total_loss += loss.item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # except:
            #     print("Skipping training batch due to an issue.")


        print("Computing Validation Loss.")
        val_total_loss = 0
        val_batches = 0
        model.eval()
        with torch.no_grad():
            for batch_num, (img_seq, x0, x_opt, u_opt, ball_future) in enumerate(val_loader):
                try:
                    val_batches += 1
                    img_seq, x0, x_opt, u_opt, ball_future = img_seq.to(device), x0.to(device), x_opt.to(device), u_opt.to(device), ball_future.to(device)

                    x_opt, u_opt, _, _ = gt_controller(x0.unsqueeze(-1),
                                                       ball_future.float())

                    # forward pass
                    img_seq = img_seq.permute(1, 0, 2, 3, 4)
                    x0 = x0.unsqueeze(-1).type(torch.float32)
                    x_sol, u_sol = model(img_seq, x0)
                    # ball_pred = model(img_seq)

                    # evaluate loss
                    loss = loss_func(u_sol, u_opt.squeeze())
                    # loss = loss_func(ball_pred, ball_future.type(torch.float32).squeeze())
                    val_total_loss += loss.item()
                except:
                    print("Skipping validation batch due to an issue.")
        model.train()

        torch.save(model, "../models/Epoch_"+str(i+1))

        print("normalized train loss: ", total_loss / num_batches)
        print("normalized val loss: ", val_total_loss / val_batches)
        train_loss.append(total_loss / num_batches)
        val_loss.append(val_total_loss / val_batches)
        np.savetxt('../task_loss/train_loss.txt', train_loss)
        np.savetxt('../task_loss/val_loss.txt', val_loss)

    return train_loss

if __name__=='__main__':
    train_options = {"epochs": 60,
                     "learning_rate": 1e-3}

    dataloader_params = {'batch_size': 32,
                         'shuffle': True,
                         'num_workers': 12,
                         'drop_last': False}

    val_dataloader_params = {'batch_size': 32,
                             'shuffle': False,
                             'num_workers': 12,
                             'drop_last': False}

    train_loss = train_model(train_options, dataloader_params, val_dataloader_params)

