import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import sys,os

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *

sns.set_theme(style="ticks", palette="pastel")

if __name__ == '__main__':

    ###############################
    results_dir = NASA_ULI_ROOT_DIR + '/pretrained_DNN/results/'

    csv_fname = results_dir + '/results.txt'

    df = pandas.read_csv(csv_fname, sep='\t')

    #x_var = 'Condition' 
    #y_var = 'Loss'
    #hue_var = 'Train/Test'

    x_var = 'condition' 
    y_var = 'loss'
    hue_var = 'model_type'

    title_str = 'Large Taxinet Pre-trained Model (ResNet-18)'

    order = ['morning', 'afternoon', 'overcast', 'night']

    # Draw a nested boxplot to show bills by day and time
    sns.barplot(x = x_var, y = y_var,
                hue = hue_var, data = df, order = order)
    plt.title(title_str)
    plt.savefig(results_dir + '/taxinet_barplot.pdf')
    plt.close()
