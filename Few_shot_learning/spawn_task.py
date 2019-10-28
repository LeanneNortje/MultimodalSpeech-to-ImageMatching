#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
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
import tensorflow as tf
import subprocess
import sys
from tqdm import tqdm

image_dataset = "../Data_processing/MNIST/test.npz"
speech_dataset = "../Data_processing/TIDigits/Subsets/Words/mfcc/gt_test_mfcc.npz"
one_shot_episode = "./Episode_files/M_11_K_1_Q_10_TIDigits_test_MNIST_test_episodes.txt"
few_shot_episode = "./Episode_files/M_11_K_5_Q_10_TIDigits_test_MNIST_test_episodes.txt"

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def main():

    fn_list = [

        ("../Model_data/cae/rnn/TIDigits/mfcc/gt/01fb7025a2/01fb7025a2_log.txt",
            "../Model_data/cae/fc/MNIST/867e802dba/867e802dba_log.txt")

        ]
    

    for speech_fn, image_fn in fn_list:
        cmd = "./few_shot_learning.py --speech_log_fn {} --image_log_fn {} --speech_data_fn {} --image_data_fn {} --episode_fn {}".format(speech_fn, image_fn, speech_dataset, image_dataset, one_shot_episode) 
        print("-"*150)
        print("\nCommand: " + cmd)
        print("\n" + "-"*150)
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()

        cmd = "./few_shot_learning.py --speech_log_fn {} --image_log_fn {} --speech_data_fn {} --image_data_fn {} --episode_fn {}".format(speech_fn, image_fn, speech_dataset, image_dataset, few_shot_episode) 
        print("-"*150)
        print("\nCommand: " + cmd)
        print("\n" + "-"*150)
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()
    
if __name__ == "__main__":
    main()