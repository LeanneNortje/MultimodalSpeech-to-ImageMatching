#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script spawns all multimodal speech-image matching tasks for all the specified speech-vision
# model pairs. If you only want to do the task for a specific pair, give the appropriate speech 
# model library and image model library and set --test_rnd_seed in .feature_learning.py to False.
# Otherwise give any model library in the respective model _log.txt file and the task will be done 
# between the speech-vison model pairs at a specific random seed. The script will match these 
# models automatically. 
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
# import tensorflow as tf
import subprocess
import sys
from tqdm import tqdm

sys.path.append("..")
from paths import model_lib_path

sys.path.append(path.join("..", model_lib_path))
import model_setup_library

image_dataset = "../Data_processing/MNIST/test.npz"
speech_dataset = "../Data_processing/TIDigits/Subsets/Words/mfcc/gt_test_mfcc.npz"
one_shot_episode = "./Episode_files/M_11_K_1_Q_10_TIDigits_test_MNIST_test_episodes.txt"
few_shot_episode = "./Episode_files/M_11_K_5_Q_10_TIDigits_test_MNIST_test_episodes.txt"

identical_keywords = ["K", "Q", "mix_training_datasets", "model_type",
"one_shot_not_few_shot", "pretrain", "shuffle_batches_every_epoch", "use_best_model", "pair_type"]

PAIRS = {"TIDigits": "MNIST", "buckeye": "omniglot"}
PRINT_LENGTH = model_setup_library.PRINT_LENGTH

#_____________________________________________________________________________________________________________________________________
#
# Function that establised if a given speech and image model are pairs for the multimodal task
#
#_____________________________________________________________________________________________________________________________________

def pair_check(speech_lib, image_lib):

    pair = True

    if speech_lib["M"] != image_lib["M"] + 1: return False

    for key in identical_keywords:

        if key not in speech_lib: speech_lib[key] = "default"
        if key not in image_lib: image_lib[key] = "default"
        
        if speech_lib[key] != image_lib[key]: pair = False
    if pair is False: return False

    if PAIRS[speech_lib["data_type"]] != image_lib["data_type"]: return False

    if speech_lib["mix_training_datasets"]:
        if PAIRS[speech_lib["other_speech_dataset"]] != image_lib["other_image_dataset"]: return False

    if speech_lib["pretrain"]:
        if speech_lib["pretraining_model"] != image_lib["pretraining_model"]: return False
        if PAIRS[speech_lib["pretraining_data"]] != image_lib["pretraining_data"]: return False
        if speech_lib["use_best_pretrained_model"] != image_lib["use_best_pretrained_model"]: return False
        if speech_lib["mix_training_datasets"]:
            if PAIRS[speech_lib["other_pretraining_speech_dataset"]] != image_lib["other_pretraining_image_dataset"]:return False

    speech_val_dataset = speech_lib["validation_speech_dataset"] if speech_lib["validate_on_validation_dataset"] else speech_lib["data_type"]
    image_val_dataset = image_lib["validation_image_dataset"] if image_lib["validate_on_validation_dataset"] else image_lib["data_type"]
    if PAIRS[speech_val_dataset] != image_val_dataset: return False

    return pair


#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def main():

    directories = os.walk("../Model_data/")
    valid_dirs = []
    for root, dirs, files in directories:
        for filename in files:
            if filename.split("_")[-1] == "log.txt"and len(dirs) != 0:
                log = path.join(root, filename)
                name = root.split("/")[-1]
                lib = path.join(root, dirs[0], name + "_lib.pkl")
                valid_dirs.append((lib, log))

    pair_logs = []
    appended_logs= []
    count = 0

    for lib1, log1 in valid_dirs:
        first_lib = model_setup_library.restore_lib(lib1)

        for lib2, log2 in valid_dirs:
            if lib1 != lib2 and log1 not in appended_logs and log2 not in appended_logs:
                second_lib = model_setup_library.restore_lib(lib2)
                
                if first_lib["training_on"] == "images": 
                    image_lib = first_lib.copy()
                    image_log = log1
                else: 
                    speech_lib = first_lib.copy()
                    speech_log = log1

                if second_lib["training_on"] == "images" and first_lib["training_on"] == "speech": 
                    image_lib = second_lib.copy()
                    image_log = log2
                elif second_lib["training_on"] == "speech" and first_lib["training_on"] == "images": 
                    speech_lib = second_lib.copy()
                    speech_log = log2
                else: break

                pairs = pair_check(speech_lib, image_lib)

                if pairs: 
                    pair_logs.append((speech_log, image_log))
                    appended_logs.append(speech_log)
                    appended_logs.append(image_log)


    for speech_fn, image_fn in pair_logs:
        cmd = "./few_shot_learning.py --speech_log_fn {} --image_log_fn {} --speech_data_fn {} --image_data_fn {} --episode_fn {}".format(speech_fn, image_fn, speech_dataset, image_dataset, one_shot_episode) 
        print_string = cmd
        model_setup_library.command_printing(print_string)
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()

        cmd = "./few_shot_learning.py --speech_log_fn {} --image_log_fn {} --speech_data_fn {} --image_data_fn {} --episode_fn {}".format(speech_fn, image_fn, speech_dataset, image_dataset, few_shot_episode) 
        print_string = cmd
        model_setup_library.command_printing(print_string)
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()
    
if __name__ == "__main__":
    main()