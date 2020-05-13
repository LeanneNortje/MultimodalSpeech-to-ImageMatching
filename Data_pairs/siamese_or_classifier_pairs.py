#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script generates speech and image positive pairs from latents extracted from a Siamese or 
# classifier model.
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
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean", "euclidean_squared"], default="cosine")
    return parser.parse_args()

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________


def main():

    args = check_argv()

    if args.metric == "cosine":
        dist_func = "cosine"
    elif args.metric == "euclidean":
        dist_func = "euclidean"
    elif args.metric == "euclidean_squared":
        dist_func == "sqeuclidean"

    print("Start time: {}".format(datetime.datetime.now()))

    model = "siamese" if args.siamese_not_classifier == "True" else "classifier"
    speech_not_image_pairs = True if args.feats_fn.split("/")[2] in SPEECH_DATASETS else False if args.feats_fn.split("/")[2] in IMAGE_DATASETS else "INVALID" 
    if speech_not_image_pairs == "INVALID":
        print("Specified dataset to get pairs for, not valid.")
        sys.exit(0)    

    key_pair_file = path.join(pair_path, "/".join(args.feats_fn.split(".")[-2].split("/")[2:]))
    util_library.check_dir(key_pair_file)
    latent_npz = path.join(key_pair_file, model + "_latents", model + "_feats.npz")
    key_pair_file = path.join(key_pair_file, "key_" + model + "_pairs.list")
    
    if os.path.isfile(latent_npz) is False:
        print("Generate latents before calculating distances.")
        sys.exit(0)

    latents, keys = data_library.load_latent_data_from_npz(latent_npz)
    latents = np.asarray(latents)
    latents = np.squeeze(latents)
  
    nan = False
    for i in range(len(latents)):
        for j in range(latents.shape[-1]):
            if np.isnan(latents[i, j]):
                nan = True

    if nan:
        print("Latents contain NAN values.")
        sys.exit(0)

    key_pair_file = open(key_pair_file, 'w')

    for i in tqdm(range(len(latents)), desc="Calculating distances", ncols=COL_LENGTH):
        
        distances = cdist(latents[i, :].reshape(1, latents.shape[-1]), latents, dist_func)
        distances = np.squeeze(distances)

        indices = np.argsort(distances)

        current_key = keys[i]

        pairs = []
        count = 0
        while len(pairs) < args.num_pairs and count < len(indices):
            pair_key = keys[indices[count]]
            
            if current_key != pair_key:
                if speech_not_image_pairs and current_key.split("_")[1].split("-")[0] != pair_key.split("_")[1].split("-")[0]: pairs.append(pair_key)
                elif speech_not_image_pairs is False:
                    pairs.append(pair_key)
            count += 1
    
        if len(pairs) == args.num_pairs:
            key_pair_file.write(f'{current_key:<30}')
            for pair in pairs:
                key_pair_file.write(f'\t{pair:<30}')
            key_pair_file.write("\n")


    key_pair_file.close()
    print("End time: {}".format(datetime.datetime.now()))

if __name__ == "__main__":
    main()