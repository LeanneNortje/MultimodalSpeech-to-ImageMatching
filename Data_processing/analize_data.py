#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script plots a histogram of the frame lengths of all the entries in a dataset. It also 
# states the mean and variance values of the overall datasets individual data entry values. 

from datetime import datetime
from os import path
import argparse
import glob
import numpy as np
import os
from os import path
from scipy.io import wavfile
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
import subprocess
import sys
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#_____________________________________________________________________________________________________________________________________
#
# Argument function
#
#_____________________________________________________________________________________________________________________________________

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_fn", type=str)
    return parser.parse_args()

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def main():

    parameters = arguments()

    feature_file = np.load(parameters.feature_fn)

    data_frames = []
    for key in feature_file:
        data_frames.append(len(feature_file[key]))

    graph_fn = path.splitext(parameters.feature_fn)[0] + "_data.jpeg"

    plt.hist(data_frames, np.arange(1, max(data_frames)+1))
    plt.xlabel("Frame lengths")
    plt.ylabel("Number of each frame length")
    plt.title("Distribution of frame lengths")
    plt.savefig(graph_fn)

    data_list = []
    for key in feature_file:
        data_list.append(feature_file[key])

    data = np.vstack(data_list)

    info_fn = path.splitext(parameters.feature_fn)[0] + "_data.txt"

    with open(info_fn, 'w') as info_file:

        info_file.write("Mean vector: {}\n".format(data[:,:].mean(axis=0)))
        info_file.write("Variance vector: {}\n".format(data[:,:].var(axis=0)))
        info_file.write("Mean: {}\n".format(data[:,:].mean()))
        info_file.write("Variance: {}\n".format(data[:,:].var()))

    info_file.close()


    
if __name__ == "__main__":
    main()