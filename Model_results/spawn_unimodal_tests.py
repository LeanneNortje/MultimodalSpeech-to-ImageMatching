#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script spawns the unimodal speech or image classification tasks for each fully trained 
# in base_dir model.
#

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

sys.path.append("..")
from paths import general_lib_path

sys.path.append(path.join("..", general_lib_path))
import util_library

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
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
    parser.add_argument("--base_dir", type=str)
    return parser.parse_args()

#_____________________________________________________________________________________________________________________________________
#
# Main 
#
#_____________________________________________________________________________________________________________________________________

def main():

    parameters = arguments()

    directories = os.walk(parameters.base_dir)
    dir_parts = parameters.base_dir.split("/")

    save_dir = path.join("./Unimodal_results", dir_parts[-4], dir_parts[-2])
    util_library.check_dir(save_dir)
    if dir_parts[-2] == "buckeye" or dir_parts[-2] == "TIDigits":
        base_dir = path.join(parameters.base_dir, "mfcc/gt/")

    lib_restore_list = []
    log_save_list = []

    for root, dirs, files in directories:
        for file in files:
            parts = path.splitext(file)
            if parts[-1] == ".pkl" and parts[0].split("_")[-1] == "lib": 
                lib_restore_list.append(path.join(root, file))
                log_save_list.append(path.join(save_dir, parts[0].split("_")[0], parts[0].split("_")[0] + "_log.txt"))

    counter = 6
    for i, lib_fn in enumerate(lib_restore_list):

        if os.path.isfile(log_save_list[i]) is False: counter = 1

        if counter <= 5:        
            cmd = "../Running_models/restore_model.py" + " --model_path {} --log_path {}".format(lib_fn, log_save_list[i])

            print("-"*150)
            print("\nCommand: " + cmd)
            print("\n" + "-"*150)
            sys.stdout.flush()
            proc = subprocess.Popen(cmd, shell=True)
            proc.wait()
            counter += 1

if __name__ == "__main__":
    main()