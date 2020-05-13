#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
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
    parser.add_argument("--all_keys_list_fn", type=str)
    parser.add_argument("--this_keys_list_fn", type=str)
    parser.add_argument("--feats_fn", type=str)
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean", "euclidean_squared"], default="cosine")
    parser.add_argument("--normalize", type=str, choices=["True", "False"], default="False")
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

    if args.metric == "cosine":
        dist_func = "cosine"
    elif args.metric == "euclidean":
        dist_func = "euclidean"
    elif args.metric == "euclidean_squared":
        dist_func == "sqeuclidean"

    print("Start time: {}".format(datetime.datetime.now()))

    num_keys = sum([1 for line in open(args.all_keys_list_fn, 'r')])
    print("Total number of keys: {}".format(num_keys))

    print("Reading images from:", args.feats_fn)
    images = np.load(args.feats_fn)
    image_dict = dict(images)

    print("Calculating distances...")
    key_pair_file = open(args.key_pair_fn, 'w')
    for line in open(args.this_keys_list_fn, 'r'):

        distances = np.zeros(num_keys)
        keys = []
        key = line.strip()

        for i_pair, this_key in enumerate(open(args.all_keys_list_fn, 'r')):
            this_key = this_key.strip()
            distances[i_pair] = cdist(image_dict[key], image_dict[this_key], dist_func)
            keys.append(this_key)

        indices = np.argsort(distances)
        key_pair_file.write("{}".format(key))
        count = 0
        i = 0
        while count < args.num_pairs:
            cur_key = keys[indices[i]]
            i += 1
            if key != cur_key: 
                key_pair_file.write(" {}".format(cur_key))
                count += 1
            if count == args.num_pairs: key_pair_file.write("\n")

    key_pair_file.close()
    print("End time: {}".format(datetime.datetime.now()))

if __name__ == "__main__":
    main()