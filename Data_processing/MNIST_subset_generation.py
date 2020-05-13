#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script downloads the MNIST dataset from Keras and splits the dataset into train, validation
# and test subsets. It also generates a unique key for each image in the MNIST dataset.  
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
import keras
import hashlib
from PIL import Image
from PIL import ImageOps

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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#_____________________________________________________________________________________________________________________________________
#
# Generating keys for the images in the MNIST dataset
#
#_____________________________________________________________________________________________________________________________________

def generate_image_lists(images, labels, file_fn=None):
    
    arrays = []
    keys = []
    lab = []
    if file_fn is not None: file = open(file_fn, "w")
    for i in range(images.shape[0]):
        
        arrays.append(images[i:i+1, :])
        hasher = hashlib.md5(repr(images[i:i+1, :]).encode("ascii"))
        key = hasher.hexdigest()
        keys.append(key)
        lab.append(str(np.argmax(labels[i:i+1, :], axis=1)[0]))

        if file_fn is not None: file.write("{} {} {}\n".format(i, lab[i], keys[i]))

    if file_fn is not None: file.close()
    return arrays, keys, lab

#_____________________________________________________________________________________________________________________________________
#
# Constructing a dictionary with the keys and correspondng images
#
#_____________________________________________________________________________________________________________________________________

def get_dict(im_x, im_keys, im_labels):
    
    image_dict ={}

    for i in range(len(im_x)):
        key = "{}_{}".format(im_labels[i], im_keys[i])
        image_dict[key] = im_x[i]

    print("Number of images in set: {}".format(len(image_dict)))
    return image_dict

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def main():

    out_dir = path.join(".", "MNIST")
    util_library.check_dir(out_dir)
    data_dir = util_library.saving_path(path.join(data_path, "MNIST"), "MNIST")
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    
    # Training data
    im_x, im_labels = mnist.train.next_batch(mnist.train.num_examples)
    im_x, im_keys, im_labels = generate_image_lists(im_x, im_labels, path.join(out_dir, "train_set_keys.txt"))
    test_im_dict = get_dict(im_x, im_keys, im_labels)
    save_fn = path.join(out_dir, "train")
    speech_library.write_feats(test_im_dict, save_fn)

    #Validation data
    im_x, im_labels = mnist.validation.next_batch(mnist.validation.num_examples)
    im_x, im_keys, im_labels = generate_image_lists(im_x, im_labels, path.join(out_dir, "validation_set_keys.txt"))
    test_im_dict = get_dict(im_x, im_keys, im_labels)
    save_fn = path.join(out_dir, "validation")
    speech_library.write_feats(test_im_dict, save_fn)

    # Testing data
    im_x, im_labels = mnist.test.next_batch(mnist.test.num_examples)
    im_x, im_keys, im_labels = generate_image_lists(im_x, im_labels, path.join(out_dir, "test_set_keys.txt"))
    test_im_dict = get_dict(im_x, im_keys, im_labels)
    save_fn = path.join(out_dir, "test")
    speech_library.write_feats(test_im_dict, save_fn)


if __name__ == "__main__":
    main()