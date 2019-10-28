#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script downsamples the images in the Omniglot dataset to 28x28 pixels. 
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
import random

sys.path.append("..")
from paths import feats_path
from paths import general_lib_path
from paths import data_path
data_path = path.join("..", data_path)

sys.path.append(path.join("..", general_lib_path))
import util_library

sys.path.append(path.join("..", feats_path))
import speech_library
feats_path = path.join("..", feats_path)

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________


def main():

    out_dir = path.join(".", "omniglot")
    util_library.check_dir(out_dir)
    data_dir = path.join(data_path, "Omniglot")
    
    
    # Testing data
    test_fn = path.join(data_dir, "test", "*", "*", "*.png")
    test_dict = {}
    save_fn = path.join(out_dir, "test")

    for im_fn in tqdm(sorted(glob.glob(test_fn))):
        im = ImageOps.invert(Image.open(im_fn).resize((28, 28)).convert('L'))
        im = np.array(im)
        im = im.reshape((1, im.shape[0]*im.shape[1]))
        im_max = np.max(im)
        im = im/im_max
        
        for row in range(im.shape[0]):
            for column in range(im.shape[1]):
                if im[row, column] > 1.0 or im[row, column] < 0.0: print("Image values not between 0 and 1")
                if np.isnan(im[row, column]): 
                    print("Image contains NaN values")
        
        name = im_fn.split("/")[-1].split(".")[0]
        test_dict[name] = im

    speech_library.write_feats(test_dict, save_fn)

    # Validation data
    train_fn = path.join(data_dir, "train", "*", "*", "*.png")
    train_dict = {}
    save_fn = path.join(out_dir, "train")

    for im_fn in tqdm(sorted(glob.glob(train_fn))):
        im = ImageOps.invert(Image.open(im_fn).resize((28, 28)).convert('L'))
        im = np.array(im)
        im = im.reshape((1, im.shape[0]*im.shape[1]))
        im_max = np.max(im)
        im = im/im_max
        
        for row in range(im.shape[0]):
            for column in range(im.shape[1]):
                if im[row, column] > 1.0 or im[row, column] < 0.0: print("Image values not between 0 and 1")
                if np.isnan(im[row, column]): 
                    print("Image contains NaN values")
        
        name = im_fn.split("/")[-1].split(".")[0]
        train_dict[name] = im

    speech_library.write_feats(train_dict, save_fn)

if __name__ == "__main__":
    main()