import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

# Create holdline network
class HoldlineNetwork(nn.Module):
    def __init__(self, img_size=(32, 64, 3), out_dim=4):
        super().__init__()

        # Get size of convolution layer output
        out_conv_size = torch.tensor([int((img_size[0] / 4) - 3), int((img_size[1] / 4) - 3), 16])

        # Define some layers
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(img_size[-1], 6, kernel_size=(5,5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5,5))
        self.dense1 = nn.Linear(torch.prod(out_conv_size), 120)
        self.dense2 = nn.Linear(120, 84)
        self.dense3 = nn.Linear(84, out_dim)

    def forward(self, x):
        # Pass through layers of NN
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        # x = F.dropout(x, p=0.1, training=True)
        x = F.relu(self.dense2(x))
        # x = F.dropout(x, p=0.1, training=True)
        out = self.dense3(x)

        # Get mean and variance
        mu = out[:, 0]
        sigma_sq = F.softplus(out[:, 1]) + torch.tensor(10e-3)
        return mu, sigma_sq

# Holdline data class
class HoldLineData(Dataset):
    def __init__(self, fname, device, channel_size=3, img_height=32, img_width=64):
        X, y = load_data(fname)
        self.X = X
        self.y = y
        self.channel_size = channel_size
        self.img_height = img_height
        self.img_width = img_width
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx].reshape((self.channel_size, self.img_height, self.img_width)).to(self.device)
        y = self.y[idx].to(self.device)

        return X, y


# Loss function for training network
def training_loss(mu, sigma_sq, y):
    lpdf = -torch.log(sigma_sq) / 2 - torch.square(y - mu) / (2 * sigma_sq)
    loss = -torch.mean(lpdf)

    return loss

# Load in training data
def load_data(fname):
    # Load HDF5 file
    data_fn = h5py.File(fname, "r")

    # Extract colored images
    imgs = np.array(data_fn["color_imgs"])

    # Extract labels
    dtps = np.array(data_fn["dtps"])

    X = torch.from_numpy(imgs).float()
    y = torch.from_numpy(-dtps).float()

    return X, y

# Train network
def train_network(epochs=10, lr=0.001, fname="hold_data.h5", batch_size=128):

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load in training data
    dataset = HoldLineData(fname, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate network
    model = HoldlineNetwork().to(device)

    # Instantiate optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Loop through epochs
    for epoch in range(epochs):

        total_loss = 0.0
        iter = 0.0
        # Loop through dataset
        for i, data in enumerate(dataloader, 0):

            # Get inputs and labels
            inputs, labels = data

            # Zero grads on optimizer
            optimizer.zero_grad()
            
            # Forward and backwards pass
            mu, sigma_sq = model(inputs)
            loss = training_loss(mu, sigma_sq, labels)
            loss.backward()
            optimizer.step()

            # Log running loss
            total_loss += loss.item()
            iter += 1.0

        total_loss = total_loss / iter

        # Print total loss
        print("Total training loss epoch " + str(epoch + 1) + ": " + str(total_loss))
            

    return model

def test_network(model, fname="hold_data.h5"):
    # Load in training data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = HoldLineData(fname, device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    predictions = np.zeros(dataset.__len__())
    labels = np.zeros(dataset.__len__())
    err = np.zeros(dataset.__len__())
    for i, data in enumerate(dataloader, 0):
        # Get inputs and labels
        img, label = data
        mu, sigma_sq = model(img)
        predictions[i] = mu.cpu().detach().numpy()
        labels[i] = label.cpu().detach().numpy()
        err[i] = sigma_sq.cpu().detach().numpy()

    return predictions, labels, err



if __name__ == "__main__":
    # Should we save the trained networks weights?
    save_weights = False

    # Train a model
    model = train_network(epochs=80)

    # Save weights if desired
    if save_weights:
        torch.save(model.state_dict(), "model_weights.pth")

    # Run the entire dataset through the trained network
    predictions, labels, err = test_network(model)

    # Look at predictions vs. labels - want to see a one-to-one correspondence
    fig, ax = plt.subplots()
    ax.errorbar(labels, predictions, yerr=err, fmt="o")
    ax.set_xlabel("True distance to holdline")
    ax.set_ylabel("Predicted distance to holdline")
    plt.show()