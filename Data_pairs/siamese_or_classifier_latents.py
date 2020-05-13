#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script generates speech or image latents from a Siamese or classifier model.
#

from __future__ import division
from __future__ import print_function
import argparse
import sys
from tqdm import tqdm
import numpy as np
import os
import datetime
from os import path
import math
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import re
import pickle
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append("..")
from paths import data_path
from paths import pair_path
from paths import data_lib_path
from paths import general_lib_path
from paths import model_lib_path
data_path = path.join("..", data_path)
pair_path = path.join("..", pair_path)

sys.path.append(path.join("..", data_lib_path))
import data_library
import batching_library

sys.path.append(path.join("..", general_lib_path))
import util_library

sys.path.append(path.join("..", model_lib_path))
import model_setup_library
import model_legos_library

SPEECH_DATASETS = model_setup_library.SPEECH_DATASETS
IMAGE_DATASETS = model_setup_library.IMAGE_DATASETS
COL_LENGTH = model_setup_library.COL_LENGTH

#_____________________________________________________________________________________________________________________________________
#
# Argument function
#
#_____________________________________________________________________________________________________________________________________


def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats_fn", type=str)
    parser.add_argument("--num_pairs", type=int)
    parser.add_argument("--siamese_not_classifier", type=str)
    return parser.parse_args()


#_____________________________________________________________________________________________________________________________________
#
# Retrieve embeddings for data inputs from specified model
#
#_____________________________________________________________________________________________________________________________________

def speech_rnn_latent_values(lib, feats_fn, siamese_not_classifier, latent_dict, ):
    
    x, labels, lengths, keys = (
        data_library.load_speech_data_from_npz(feats_fn)
        )

    data_library.truncate_data_dim(x, lengths, lib["input_dim"], lib["max_frames"])
    iterator = batching_library.speech_iterator(
        x, 1, shuffle_batches_every_epoch=False
        )
    indices = iterator.indices

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    key_counter = 0

    if siamese_not_classifier:
        X = tf.placeholder(tf.float32, [None, None, lib["input_dim"]])
        target = tf.placeholder(tf.int32, [None])
        X_lengths = tf.placeholder(tf.int32, [None])
        train_flag = tf.placeholder_with_default(False, shape=())
        training_placeholders = [X, X_lengths, target]
        
        model = model_legos_library.siamese_rnn_architecture(
            [X, X_lengths], train_flag, model_setup_library.activation_lib(), lib
            )

        latent = model["output"]
        output = latent
    else:
        X = tf.placeholder(tf.float32, [None, None, lib["input_dim"]])
        target = tf.placeholder(tf.int32, [None, lib["num_classes"]])
        X_lengths = tf.placeholder(tf.int32, [None])
        train_flag = tf.placeholder_with_default(False, shape=())
        training_placeholders = [X, X_lengths, target]
        
        model = model_legos_library.rnn_classifier_architecture(
            [X, X_lengths], train_flag, model_setup_library.activation_lib(), lib
            )

        output = model["output"]
        latent = model["latent"]

    model_fn = model_setup_library.get_model_fn(lib)

    saver = tf.train.Saver()
    with tf.Session(config=config) as sesh:
        saver.restore(sesh, model_fn)
        for feats, lengths in tqdm(iterator, desc="Extracting latents", ncols=COL_LENGTH):
            lat = sesh.run(
                latent, feed_dict={X: feats, X_lengths: lengths, train_flag: False}
                )

            latent_dict[keys[indices[key_counter]]] = lat
            key_counter += 1

    print("Total number of keys: {}".format(key_counter))


def image_cnn_latent_values(lib, feats_fn, siamese_not_classifier, latent_dict):

    x, labels, keys = (
        data_library.load_image_data_from_npz(
            feats_fn
            )
        )

    iterator = batching_library.unflattened_image_iterator(
        x, 1, shuffle_batches_every_epoch=False
        )

    indices = iterator.indices

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    key_counter = 0

    if siamese_not_classifier:
        X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        target =  tf.placeholder(tf.float32, [None])
        train_flag = tf.placeholder_with_default(False, shape=())

        model = model_legos_library.siamese_cnn_architecture(
            X, train_flag, lib["enc"], lib["enc_strides"], model_setup_library.pooling_lib(), lib["pool_layers"], lib["latent"], 
            lib, model_setup_library.activation_lib(), print_layer=True
            )

        latent = tf.nn.l2_normalize(model["output"], axis=1)
        output = latent
    else:
        X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        target =  tf.placeholder(tf.float32, [None, lib["num_classes"]])
        train_flag = tf.placeholder_with_default(False, shape=())

        model = model_legos_library.cnn_classifier_architecture(
            X, train_flag, lib["enc"], lib["enc_strides"], model_setup_library.pooling_lib(), lib["pool_layers"], lib["latent"], 
            lib, model_setup_library.activation_lib(), print_layer=True
            )

        output = model["output"]
        latent = model["latent"]
    
    model_fn = model_setup_library.get_model_fn(lib)

    saver = tf.train.Saver()
    with tf.Session(config=config) as sesh:
        saver.restore(sesh, model_fn)
        for feats in tqdm(iterator, desc="Extracting latents", ncols=COL_LENGTH):
            lat = sesh.run(
                latent, feed_dict={X: feats, train_flag: False}
                )
            latent_dict[keys[indices[key_counter]]] = lat
            key_counter += 1

    print("Total number of keys: {}".format(key_counter))

    

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________


def main():

    args = check_argv()

    print("Start time: {}".format(datetime.datetime.now()))

    model = "siamese" if args.siamese_not_classifier == "True" else "classifier"
    speech_not_image_pairs = True if args.feats_fn.split("/")[2] in SPEECH_DATASETS else False if args.feats_fn.split("/")[2] in IMAGE_DATASETS else "INVALID" 
    if speech_not_image_pairs == "INVALID":
        print("Specified dataset to get pairs for, not valid.")
        sys.exit(0)

    VALID_DATASETS = SPEECH_DATASETS if speech_not_image_pairs else IMAGE_DATASETS
    # VALID_DATASETS = ["buckeye"] if speech_not_image_pairs else ["omniglot"]

    directories = os.walk("../Model_data/")
    valid_dirs = []

    for root, dirs, files in directories:
        for filename in files:
            if filename.split("_")[-1] == "log.txt" and root.split("/")[2] == model and root.split("/")[4] in VALID_DATASETS:
                log = path.join(root, filename)
                name = root.split("/")[-1]
                valid_dirs.append((log, root, root.split("/")[-1]))

    if len(valid_dirs) != 1:
        print(f'Number of models found to generate pairs is {len(valid_dirs)}, can only generate pairs from 1 model')
        sys.exit(0)

    acc_dict = {}
    for line in open(valid_dirs[0][0], 'r'):
        if re.search("rnd_seed", line):
            line_parts = line.strip().split(" ")
            keyword_parts = "at rnd_seed of ".strip().split(" ")
            ind = np.where(np.asarray(line_parts) == keyword_parts[0])[0]
            if ":".join(line_parts[0].split(":")[:-1]) not in acc_dict:
                acc_dict[":".join(line_parts[0].split(":")[:-1])] = (float(line_parts[ind[0]-1]) + float(line_parts[ind[1]-1]))/2.0
            max_acc = -np.inf 
            max_name = ""
            for name in acc_dict: 
                if acc_dict[name] > max_acc:
                    max_acc = acc_dict[name]
                    max_name = name
    
    model_path = path.join(valid_dirs[0][1], max_name, valid_dirs[0][2] + "_lib.pkl")
    key_pair_file = path.join(pair_path, "/".join(args.feats_fn.split(".")[-2].split("/")[2:]), model + "_latents")
    util_library.check_dir(key_pair_file)
    key_pair_file = path.join(key_pair_file, model + "_feats.npz")
    latent_dict = {}

    if os.path.isfile(key_pair_file) is False:
    
        if speech_not_image_pairs:
            print(f'Restoring model from: {model_path}')
            speech_lib = model_setup_library.restore_lib(model_path)

            latents = speech_rnn_latent_values(speech_lib, args.feats_fn, args.siamese_not_classifier, latent_dict)

        else: 
            print(f'Restoring model from: {model_path}')
            image_lib = model_setup_library.restore_lib(model_path)   

            latents = image_cnn_latent_values(image_lib, args.feats_fn, args.siamese_not_classifier, latent_dict)


        
        np.savez_compressed(key_pair_file, **latent_dict)
        
    print("End time: {}".format(datetime.datetime.now()))

if __name__ == "__main__":
    main()