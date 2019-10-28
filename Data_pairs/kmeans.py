#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script calculates genrates image pairs by using k-means to get the N closest images to the 
# query. 
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats_fn", type=str)
    parser.add_argument("--key_pair_fn", type=str)
    parser.add_argument("--num_pairs", type=int)
    return parser.parse_args()

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________


def main():

    args = check_argv()

    print("Start time: {}".format(datetime.datetime.now()))

    x, labels, keys = (
        data_library.load_image_data_from_npz(
            args.feats_fn
            )
        )

    print("Total number of keys: {}".format(len(keys)))

    x_array = np.empty((len(x), x[0].shape[-1]))
    for i in range(len(x)):
        x_array[i, :] = x[i]

    print("Calculating distances...")
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(x_array)
    distances, indices = nbrs.kneighbors(x_array)
    key_pair_file = open(args.key_pair_fn, 'w')

    for i in range(len(indices)):

        key_pair_file.write("{}".format(keys[i]))
        count = 0
        j = 0
        while count < args.num_pairs:
            cur_key = keys[indices[i, j]]
            j += 1
            if keys[i] != cur_key: 
                key_pair_file.write(" {}".format(cur_key))
                count += 1
            if count == args.num_pairs: key_pair_file.write("\n")

    key_pair_file.close()
    print("End time: {}".format(datetime.datetime.now()))

if __name__ == "__main__":
    main()