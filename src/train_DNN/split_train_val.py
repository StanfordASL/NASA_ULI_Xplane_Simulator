import os, sys
import numpy as np

from shutil import copyfile

NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']
UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *

SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

data_dir = NASA_ULI_ROOT_DIR + '/data/'
src_data_dir = data_dir + 'nominal_conditions/'

if __name__ == '__main__':

    filename_list = os.listdir(src_data_dir)
    png_files = [files for files in filename_list if files.endswith('.png')]
    episodes_list = list(set([x.split('_')[4] for x in png_files]))

    num_train_episodes = 12
    train_start = 0
    train_end = num_train_episodes
    num_test_episodes = 4
    test_start = train_end
    test_end = train_end + num_test_episodes
    num_val_episodes = 4
    val_start = test_end
    val_end = test_end + num_val_episodes

    shuffled_episodes = np.random.permutation(episodes_list)
    train_episodes = shuffled_episodes[train_start:train_end]
    test_episodes = shuffled_episodes[test_start:test_end]
    val_episodes = shuffled_episodes[val_start:val_end]

    print("train_episodes: ", train_episodes)
    print("test_episodes: ", test_episodes)
    print("val_episodes: ", val_episodes)

    train_dir = remove_and_create_dir(data_dir + 'nominal_conditions_train/')
    test_dir = remove_and_create_dir(data_dir + 'nominal_conditions_test/')
    val_dir = remove_and_create_dir(data_dir + 'nominal_conditions_val/')

    for image in png_files:
        episode_num = image.split('_')[4]
        if episode_num in train_episodes:
            copyfile(src_data_dir + image, train_dir + image)
        elif episode_num in test_episodes:
            copyfile(src_data_dir + image, test_dir + image)
        elif episode_num in val_episodes:
            copyfile(src_data_dir + image, val_dir + image)
        else:
            print("Unhandled case.")
