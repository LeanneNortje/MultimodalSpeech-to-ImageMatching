#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
# Some fragment of code adapted from and credit given to: Herman Kamper
#_________________________________________________________________________________________________
#
# This script calculates the distances between speech features with DTW. The DTW script is directly 
# used from the speech_dtw repo written by Herman Kamper. 
#

from __future__ import division
from __future__ import print_function
from os import path
import argparse
import os
import datetime
import numpy as np
import sys
import time

sys.path.append("..")
from paths import src
from paths import speech

sys.path.append(path.join("..", src, speech))

from speech_dtw import _dtw

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

#_____________________________________________________________________________________________________________________________________
#
# Argument function
#
#_____________________________________________________________________________________________________________________________________

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_fn", type=str)
    parser.add_argument("--feats_fn", type=str)
    parser.add_argument("--distances_fn", type=str)
    parser.add_argument("--binary_distances", type=str, choices=["True", "False"], default="False")
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean", "euclidean_squared"], default="cosine")
    parser.add_argument("--normalize", type=str, choices=["True", "False"], default="False")
    return parser.parse_args()

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def main():

    args = check_argv()

    if args.metric == "cosine":
        dist_func = _dtw.multivariate_dtw_cost_cosine
    elif args.metric == "euclidean":
        dist_func = _dtw.multivariate_dtw_cost_euclidean
    elif args.metric == "euclidean_squared":
        dist_func == _dtw.multivariate_dtw_cost_euclidean_squared

    print("Start time: {}".format(datetime.datetime.now()))

    print("Reading pairs from: ", args.pairs_fn)
    pairs = []
    for line in open(args.pairs_fn):
        utt1, utt2 = line.split()
        pairs.append((utt1, utt2))

    print("Reading features from:", args.feats_fn)
    feats = np.load(args.feats_fn)
    feats = dict(feats)

    if args.normalize == "True":
        print("Normalizing features...")
        for utt in feats:
            N = feats[utt].shape[0]
            for i in range(N):
                feats[utt][i, :] = feats[utt][i, :]/np.linalg.norm(feats[utt][i, :])

    print("Calculating distances...")
    distances = np.zeros(len(pairs))
    for i_pair, pair in enumerate(pairs):
        utt1, utt2 = pair
        distances[i_pair] = dist_func(
            np.array(feats[utt1], dtype=np.double), np.array(feats[utt2], dtype=np.double), True
            )

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