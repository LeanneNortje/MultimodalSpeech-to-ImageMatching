#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script calculates the cosine distance between images.
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
import generate_episodes

sys.path.append("..")
from paths import data_path
from paths import data_lib_path
from paths import general_lib_path
data_path = path.join("..", data_path)

sys.path.append(path.join("..", data_lib_path))
import data_library

sys.path.append(path.join("..", general_lib_path))
import util_library

#_____________________________________________________________________________________________________________________________________
#
# Argument function
#
#_____________________________________________________________________________________________________________________________________

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_fn", type=str)
    parser.add_argument("--feats_fn", type=str)
    parser.add_argument("--distances_fn", type=str)
    parser.add_argument("--binary_distances", type=str, choices=["True", "False"], default="False")
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

    print("Reading pairs from: ", args.pairs_fn)
    pairs = []
    for line in open(args.pairs_fn):
        key1, key2 = line.split()
        pairs.append((key1, key2))

    print("Reading images from:", args.feats_fn)
    images = np.load(args.feats_fn)
    image_dict = dict(images)

    print("Calculating distances...")
    distances = np.zeros(len(pairs))
    for i_pair, pair in enumerate(pairs):
        key1, key2 = pair
        distances[i_pair] = cdist(image_dict[key1], image_dict[key2], dist_func)

    if args.binary_distances == "True":
        print("Writing distances to binary file: {}".format(args.distances_fn))
        np.asarray(distances, dtype=np.float32).tofile(args.distances_fn)
    else:
        print("Writing distances to text file: {}".format(args.distances_fn))
        np.asarray(distances, dtype=np.float32).tofile(args.distances_fn, "\n")
        open(args.distances_fn, "a").write("\n")
    print("End time: {}".format(datetime.datetime.now()))

if __name__ == "__main__":
    main()