import sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def basic_plot_ts(ts_vector, plot_file, title_str = None, ylabel = None, lw=3.0, ylim = None, xlabel = 'time'):
    plt.plot(ts_vector, lw=lw)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if ylim:
        plt.ylim(ylim[0], ylim[1])

    plt.title(title_str)
    plt.savefig(plot_file)
    plt.close()

