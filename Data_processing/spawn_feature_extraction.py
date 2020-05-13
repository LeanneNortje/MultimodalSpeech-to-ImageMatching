#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script simply spawns each speech feature extraction that we will need. 
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
import timeit

sys.path.append("..")
from paths import model_lib_path
sys.path.append(path.join("..", model_lib_path))
import model_setup_library

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PRINT_LENGTH = model_setup_library.PRINT_LENGTH

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def main():

    feats_commands = [
        "--dataset TIDigits --feat_type mfcc",
        "--dataset TIDigits --feat_type fbank",
        "--dataset buckeye --feat_type mfcc",
        "--dataset buckeye --feat_type fbank"
        ]
    
    start = timeit.default_timer() 
    for this_command in feats_commands:
        cmd = "./feature_extraction.py " + this_command
        print("-"*150)
        print("\nCommand: " + cmd)
        print("\n" + "-"*PRINT_LENGTH)
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()

    end = timeit.default_timer() 
    total_time = end - start
    hours = int(int(total_time/60)/60)
    minutes = int((total_time - (hours*60*60))/60)
    seconds = (total_time - (hours*60*60) - (minutes*60))

    print("./spawn_feature_extraction.py took {}hrs {}min {}sec to extract all features.".format(hours, minutes, seconds))
    
if __name__ == "__main__":
    main()