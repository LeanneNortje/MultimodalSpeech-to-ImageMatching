#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
#
# This script spawns all speech and image positive pair generation for all specified speech and image 
# datasets and subsets. 
#

from __future__ import division
from __future__ import print_function
import argparse
import sys
import numpy as np
import os
import datetime
from os import path
import math
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import re
import itertools
import subprocess

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append("..")
from paths import data_path
from paths import pair_path
from paths import feats_path
from paths import data_lib_path
from paths import general_lib_path
from paths import model_lib_path
data_path = path.join("..", data_path)
pair_path = path.join("..", pair_path)
feats_path = path.join("..", feats_path)

sys.path.append(path.join("..", data_lib_path))
import data_library
import batching_library

sys.path.append(path.join("..", general_lib_path))
import util_library

sys.path.append(path.join("..", model_lib_path))
import model_setup_library
import model_legos_library

SPEECH_DATASETS = model_setup_library.SPEECH_DATASETS
SPEECH_DATASET_TYPE = ["train", "val", "test"]
IMAGE_DATASETS = model_setup_library.IMAGE_DATASETS
IMAGE_DATASET_TYPE = ["train", "validation", "test"]
NUM_PAIRS = [5]
METRIC = ["cosine"]
pair_generation_model = ["classifier"]

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________


def main():

    feat_fns = []

    for dataset_type in SPEECH_DATASET_TYPE:
        for dataset in SPEECH_DATASETS:
            feat_fns.append(path.join(feats_path, dataset, "Subsets", "Words", "mfcc", "gt_" + dataset_type + "_mfcc.npz"))
    

    for dataset_type in IMAGE_DATASET_TYPE:
        for dataset in IMAGE_DATASETS:
            feat_fns.append(path.join(feats_path, dataset, dataset_type + ".npz"))

    for (fn, num, model, metr) in list(itertools.product(feat_fns, NUM_PAIRS, [model == "siamese" for model in pair_generation_model], METRIC)):  
        cmd = "./siamese_or_classifier_latents.py " + " --feats_fn {} --num_pairs {} --siamese_not_classifier {}".format(fn, num, model)
        print_string = cmd
        model_setup_library.command_printing(print_string)
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()
        
        cmd = "./siamese_or_classifier_pairs.py " + " --feats_fn {} --num_pairs {} --siamese_not_classifier {} --metric {}".format(fn, num, model, metr)
        print_string = cmd
        model_setup_library.command_printing(print_string)
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()

if __name__ == "__main__":
    main()