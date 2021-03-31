"""
    Goal: Visualize images from aircraft camera and load as a pytorch dataloader
    0. load images and the corresponding state information in labels.csv
    1. test a trained DNN and visualize predictions
"""

import sys, os
import torch
import numpy as np
import pandas
import matplotlib.pyplot as plt

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']

import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

DATA_DIR = NASA_ULI_ROOT_DIR + '/data/'

CODE_DIR = NASA_ULI_ROOT_DIR + '/src/train_DNN/'
sys.path.append(CODE_DIR)

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *
from model_taxinet import TaxiNetDNN

if __name__ == '__main__':

    # CUDA model
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('found device: ', device)

    model_dir = NASA_ULI_ROOT_DIR + '/model'

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

    # LOSS FUNCTION
    loss_func = torch.nn.MSELoss().to(device)

    # how often to plot a few images for progress report
    # warning: plotting is slow
    NUM_PRINT = 2

    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224

    # create a temp dir to visualize a few images
    visualization_dir = SCRATCH_DIR + '/test_DNN_taxinet/viz/'
    remove_and_create_dir(visualization_dir)

    MAX_FILES = 200

    # where original XPLANE images are stored 
    data_dir = DATA_DIR + '/nominal_conditions_val/'
   
    # resize to 224 x 224 x 3 for EfficientNets
    # prepare image transforms
    # warning: you might need to change the normalization values given your dataset's statistics
    tfms = transforms.Compose([transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]),])

    image_list = [x for x in os.listdir(data_dir) if x.endswith('.png')]

    # where the labels for each image (such as distance to centerline) are present
    label_file = data_dir + '/labels.csv'
 
    # dataframe of labels
    labels_df = pandas.read_csv(label_file, sep=',')
    
    # columns are: 
    # ['image_filename', 'absolute_time_GMT_seconds', 'relative_time_seconds', 'distance_to_centerline_meters', 'distance_to_centerline_NORMALIZED', 'downtrack_position_meters', 'downtrack_position_NORMALIZED', 'heading_error_degrees', 'heading_error_NORMALIZED', 'period_of_day', 'cloud_type']

    for i, image_name in enumerate(image_list):

        # open images and apply transforms
        fname = data_dir + '/' + str(image_name)
        image = Image.open(fname).convert('RGB')
        tensor_image_example = tfms(image)

        # get the corresponding state information (labels) for each image
        specific_row = labels_df[labels_df['image_filename'] == image_name]
        # there are many states of interest, you can modify to access which ones you want
        dist_centerline_norm = specific_row['distance_to_centerline_NORMALIZED'].item()
        # normalized downtrack position
        downtrack_position_norm = specific_row['downtrack_position_NORMALIZED'].item()

        # normalized heading error
        heading_error_norm = specific_row['heading_error_NORMALIZED'].item()

        labels = torch.tensor([dist_centerline_norm, downtrack_position_norm])

        # run the model and get the loss
        inputs = tensor_image_example.unsqueeze(0)
        #inputs = tensor_image_example
        inputs = inputs.to(device)
        labels = labels.to(device)

        print(' ')
        print('inputs: ', inputs.shape)
        print('labels: ', labels)
        outputs = model(inputs)
        loss = loss_func(outputs, labels).item()
        
        preds = outputs.detach().cpu().numpy()[0]
        print('preds: ', preds)
        print('loss: ', loss)
        print(' ')

        # periodically save the images to disk 
        if i % NUM_PRINT == 0:
            plt.imshow(image)
            # original image
            title_str_1 = ' '.join(['TRUE Dist Centerline: ', str(round(dist_centerline_norm,3)), 'Downtrack Pos. Norm: ', str(round(downtrack_position_norm,3)), '\n', 'Heading Error Norm: ', str(round(heading_error_norm, 3))])

            title_str_2 = ' '.join(['PRED Dist Centerline: ', str(round(preds[0],3)), 'Downtrack Pos. Norm: ', str(round(preds[1],3))])

            title_str = title_str_1 + '\n' + title_str_2
            plt.title(title_str)
            plt.savefig(visualization_dir + '/' + str(i) + '.png')
            plt.close()

        # early terminate for debugging
        if i > MAX_FILES:
            break

