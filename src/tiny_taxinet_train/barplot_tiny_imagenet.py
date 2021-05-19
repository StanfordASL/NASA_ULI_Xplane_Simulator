import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import sys, os

sns.set_theme(style="ticks", palette="pastel")

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']

DATA_DIR = os.environ['NASA_DATA_DIR'] 

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)

from textfile_utils import *

if __name__=='__main__':

    # where the training results should go
    results_dir = SCRATCH_DIR + '/test_tiny_taxinet_DNN/'

    ###############################
    csv_fname = results_dir + '/results.txt'

    df = pandas.read_csv(csv_fname, sep='\t')

    x_var = 'Evaluation Condition' 
    y_var = 'Loss'
    hue_var = 'Train Condition'

    title_str = 'Tiny Taxinet Pytorch Model'

    order = ['morning', 'afternoon', 'overcast', 'night']

    # barplot 1
    #########################################################
    # Draw a nested boxplot to show bills by day and time
    sns.boxplot(x = x_var, y = y_var,
                hue = hue_var, data = df, order = order)
    plt.title(title_str)
    plt.savefig(results_dir + '/all_tiny_taxinet_barplot.pdf')
    plt.close()

    # barplot 2
    #########################################################
    order = ['morning', 'afternoon', 'overcast']

    # Draw a nested boxplot to show bills by day and time
    sns.boxplot(x = x_var, y = y_var,
                hue = hue_var, data = df, order = order)
    plt.title(title_str)
    plt.savefig(results_dir + '/subset_tiny_taxinet_barplot.pdf')
    plt.close()

    x_var = 'Evaluation Condition' 
    y_var = 'Loss'
    hue_var = 'Model Type'

    title_str = 'Tiny Taxinet Pytorch Model'

    order = ['morning', 'afternoon', 'overcast', 'night']

    col_var = 'Train Condition'
    # barplot 1
    #########################################################
    # Draw a nested boxplot to show bills by day and time
    sns.catplot(x = x_var, y = y_var, col = hue_var, 
                hue = col_var, data = df, order = order, kind='bar')
    plt.savefig(results_dir + '/type_tiny_taxinet_barplot.pdf')
    plt.close()

