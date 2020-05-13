#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script splits the Buckeye dataset into train, validation and test subsets and removes any
# digit classes from the subsets. 
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
import PIL
from PIL import Image
from PIL import ImageOps
import logging
import tensorflow as tf
import hashlib

import subprocess
import sys
from tqdm import tqdm
import shutil
import random

sys.path.append("..")
from paths import data_lib_path
from paths import feats_path
from paths import general_lib_path
from paths import data_path
from paths import model_lib_path
data_path = path.join("..", data_path)

sys.path.append(path.join("..", general_lib_path))
import util_library

sys.path.append(path.join("..", data_lib_path))
import data_library

sys.path.append(path.join("..", feats_path))
import speech_library
feats_path = path.join("..", feats_path)

sys.path.append(path.join("..", model_lib_path))
import model_setup_library

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def main():

    out_dir = path.join(".", "buckeye")
    util_library.check_dir(out_dir)
    data_dir = path.join(".", "buckeye", "Subsets", "Words", "mfcc")
    util_library.check_dir(path.join(data_dir, "back_up"))
    
    train_file = path.join(data_path, "buckeye", "train_classes.list")
    val_file = path.join(data_path, "buckeye", "validation_classes.list")
    test_file = path.join(data_path, "buckeye", "test_classes.list")

    train_fn = path.join(data_dir, "gt_train_mfcc.npz")
    val_fn = path.join(data_dir, "gt_val_mfcc.npz")
    test_fn = path.join(data_dir, "gt_test_mfcc.npz")

    train_dict = {}
    val_dict = {}
    test_dict = {}

    train_x, train_labels, train_lengths, train_keys = (
        data_library.load_speech_data_from_npz(train_fn)
        )
    train_x, train_labels, train_lengths, train_keys = (
        data_library.remove_test_classes(
            train_x, train_labels, train_lengths, train_keys, model_setup_library.DIGIT_LIST
            )
        )
    data_library.test_classes(train_labels, model_setup_library.DIGIT_LIST, "training")

    val_x, val_labels, val_lengths, val_keys = (
        data_library.load_speech_data_from_npz(val_fn)
        )
    val_x, val_labels, val_lengths, val_keys = (
        data_library.remove_test_classes(
            val_x, val_labels, val_lengths, val_keys, model_setup_library.DIGIT_LIST
            )
        )
    data_library.test_classes(val_labels, model_setup_library.DIGIT_LIST, "valdiation")

    test_x, test_labels, test_lengths, test_keys = (
        data_library.load_speech_data_from_npz(test_fn)
        )
    test_x, test_labels, test_lengths, test_keys = (
        data_library.remove_test_classes(
            test_x, test_labels, test_lengths, test_keys, model_setup_library.DIGIT_LIST
            )
        )
    data_library.test_classes(test_labels, model_setup_library.DIGIT_LIST, "testing")

    train_classes = []
    val_classes = []
    test_classes = []

    for line in open(train_file):
        train_classes.append(line.strip())

    for line in open(val_file):
        val_classes.append(line.strip())

    for line in open(test_file):
        test_classes.append(line.strip())

    for i, lab in enumerate(train_labels):
        if lab in train_classes: train_dict[train_keys[i]] = train_x[i]
        elif lab in val_classes: val_dict[train_keys[i]] = train_x[i]
        elif lab in test_classes: test_dict[train_keys[i]] = train_x[i]
        else: print(f'{lab} does not fit in anywhere :(')

    for i, lab in enumerate(val_labels):
        if lab in train_classes: train_dict[val_keys[i]] = val_x[i]
        elif lab in val_classes: val_dict[val_keys[i]] = val_x[i]
        elif lab in test_classes: test_dict[val_keys[i]] = val_x[i]
        else: print(f'{lab} does not fit in anywhere :(')

    for i, lab in enumerate(test_labels):
        if lab in train_classes: train_dict[test_keys[i]] = test_x[i]
        elif lab in val_classes: val_dict[test_keys[i]] = test_x[i]
        elif lab in test_classes: test_dict[test_keys[i]] = test_x[i]
        else: print(f'{lab} does not fit in anywhere :(')


    speech_library.write_feats(train_dict, train_fn)
    speech_library.write_feats(val_dict, val_fn)
    speech_library.write_feats(test_dict, test_fn)

if __name__ == "__main__":
    main()