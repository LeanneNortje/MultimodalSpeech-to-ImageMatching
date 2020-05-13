#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script copies all the non-final speech or image models' unimodal and multiodal few-shot 
# tests. 
#

from datetime import datetime
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
import shutil
import re

sys.path.append("..")
from paths import general_lib_path
from paths import model_lib_path

sys.path.append(path.join("..", model_lib_path))
import model_setup_library

sys.path.append(path.join("..", general_lib_path))
import util_library

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():

    directories = os.walk("../Model_data_non_final/")
    keyword = " at rnd_seed of "
    log_restore_list = []
    log_save_dict = {}
    model_log = []
    for root, dirs, files in directories:
        for filename in files:
            if filename.split("_")[-1] == "log.txt":
                log = path.join(root, filename)
                log_restore_list.append(log)


    for directory in log_restore_list:

        dir_parts = directory.split("/")
        save_dir = path.join("./Non_final_unimodal_results", "/".join(dir_parts[2:5] + dir_parts[-2:]))
        util_library.check_dir(path.dirname(save_dir))

        log_save_dict[directory] = save_dir

    for log_fn in log_save_dict:
  
        all_model_instances = []

        if os.path.isfile(log_fn):
            for line in open(log_fn, 'r'):

                if re.search(keyword, line):
                    line_parts = line.strip().split(" ")
                    all_model_instances.append(":".join(line_parts[0].split(":")[0:-1]))

        if os.path.isfile(log_save_dict[log_fn]) is False and len(all_model_instances) in [3, 5]: 
            print(f'\tCopying {log_fn} to {log_save_dict[log_fn]}.\n')       
            shutil.copyfile(log_fn, log_save_dict[log_fn])

        elif os.path.isfile(log_save_dict[log_fn]) is False and len(all_model_instances) == 0:
            print(f'\t{log_fn} had no successfully trained and tested models.\n')

        elif os.path.isfile(log_save_dict[log_fn]) is False and len(all_model_instances) not in [3, 5]:
            print(f'\tNot all instances in {log_fn} trained.\n')

        elif os.path.isfile(log_save_dict[log_fn]):
            print(f'\t{log_fn} already copied.\n')
    
if __name__ == "__main__":
    main()