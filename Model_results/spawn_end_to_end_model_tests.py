#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script copies all the speech-image models' unimodal and multiodal few-shot tests. 
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
import shutil
import re

sys.path.append("..")
from paths import general_lib_path
from paths import model_lib_path

sys.path.append(path.join("..", model_lib_path))
import model_setup_library

sys.path.append(path.join("..", general_lib_path))
import util_library

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
end_to_end_models = ["sv_cae", "sv_siamese_like"]

def main():

    directories = os.walk("../Model_data/")
    keyword = " at rnd_seed of "
    log_restore_list = []
    log_save_dict = {}
    zero_shot_format = {}
    model_log = []
    for root, dirs, files in directories:
        for filename in files:
            if filename.split("_")[-1] == "log.txt" and root.split("/")[2] in end_to_end_models:
                log = path.join(root, filename)
                log_restore_list.append(log)


    for directory in log_restore_list:
        
        dir_parts = directory.split("/")
        filename = dir_parts[-1]
        if filename.split("_")[1] == "unimodal" and filename.split("_")[2] == "speech": 
            save_dir = path.join("./Unimodal_results", dir_parts[2], "unimodal_speech", dir_parts[3])
            util_library.check_dir(save_dir)
            log_save_dict[directory] = [path.join(save_dir, "1_shot_results.txt"), path.join(save_dir, "5_shot_results.txt")]
            zero_shot_format[directory] = False
        elif filename.split("_")[1] == "unimodal" and filename.split("_")[2] == "image":
            save_dir = path.join("./Unimodal_results", dir_parts[2], "unimodal_image", dir_parts[3])
            util_library.check_dir(save_dir)
            log_save_dict[directory] = [path.join(save_dir, "1_shot_results.txt"), path.join(save_dir, "5_shot_results.txt")]
            zero_shot_format[directory] = False
        elif filename.split("_")[1] == "multimodal":
            save_dir = path.join("./Multimodal_results", dir_parts[2] + "_" + dir_parts[3])
            util_library.check_dir(save_dir)
            if filename.split(".")[0].split("_")[2] == "zero": 
                log_save_dict[directory] = [path.join(save_dir, "0_shot_results.txt")]
                zero_shot_format[directory] = True
            else: 
                log_save_dict[directory] = [path.join(save_dir, "1_shot_results.txt"), path.join(save_dir, "5_shot_results.txt")]
                zero_shot_format[directory] = False
    
    for log_fn in log_save_dict:

        all_model_instances = []

        if os.path.isfile(log_fn):
            for line in open(log_fn, 'r'):

                if re.search(keyword, line):
                    line_parts = line.strip().split(" ")
                    all_model_instances.append(":".join(line_parts[0].split(":")[0:-1]))

        if zero_shot_format[log_fn] is False:

            if os.path.isfile(log_save_dict[log_fn][0]) is False and os.path.isfile(log_save_dict[log_fn][1]) is False and len(all_model_instances) in [3, 5]: 

                if os.path.isfile(log_fn):
                    one_shot_text = ""
                    few_shot_text = ""
                    for line in open(log_fn, 'r'): 
                        if re.search(keyword, line):
                            line_parts = line.strip().split(" ")    
                            one_shot_text += "1-shot accuracy of " + line_parts[4] + " at rnd_seed of " + line_parts[8] + "\n"
                            few_shot_text += "5-shot accuracy of " + line_parts[-5] + " at rnd_seed of " + line_parts[-1] + "\n"
                        else:
                            one_shot_text += line
                            few_shot_text += line

                    with open(log_save_dict[log_fn][0], "w") as f:
                        f.write(one_shot_text)
                        f.close()
                    with open(log_save_dict[log_fn][1], "w") as f:
                        f.write(few_shot_text)
                        f.close()

        elif zero_shot_format[log_fn]:

            if os.path.isfile(log_save_dict[log_fn][0]) is False and len(all_model_instances) in [3, 5]:

                if os.path.isfile(log_fn):
                    zero_shot_text = ""
                    for line in open(log_fn, 'r'): 
                        if re.search(keyword, line):
                            line_parts = line.strip().split(" ")    
                            zero_shot_text += "Zero-shot accuracy of " + line_parts[4] + " at rnd_seed of " + line_parts[8] + "\n"
                        else:
                            zero_shot_text += line

                    with open(log_save_dict[log_fn][0], "w") as f:
                        f.write(zero_shot_text)
                        f.close()

        elif os.path.isfile(log_save_dict[log_fn][0]) is False and os.path.isfile(log_save_dict[log_fn][1]) is False and len(all_model_instances) == 0:
            print(f'\t{log_fn} had no successfully trained and tested models.\n')

        elif os.path.isfile(log_save_dict[log_fn][0]) is False and os.path.isfile(log_save_dict[log_fn][1]) is False and len(all_model_instances) not in [3, 5]:
            print(f'\tNot all instances in {log_fn} trained.\n')

        elif os.path.isfile(log_save_dict[log_fn][0]) and os.path.isfile(log_save_dict[log_fn][1]):
            print(f'\t{log_fn} already copied.\n')
           
if __name__ == "__main__":
    main()