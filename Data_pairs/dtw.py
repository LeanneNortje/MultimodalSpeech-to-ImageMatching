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

    dataset = args.feats_fn.strip().split("/")[2]


    if args.metric == "cosine":
        dist_func = _dtw.multivariate_dtw_cost_cosine
    elif args.metric == "euclidean":
        dist_func = _dtw.multivariate_dtw_cost_euclidean
    elif args.metric == "euclidean_squared":
        dist_func == _dtw.multivariate_dtw_cost_euclidean_squared

    print("Start time: {}".format(datetime.datetime.now()))

    num_keys = sum([1 for line in open(args.all_keys_list_fn, 'r')])
    print("Total number of keys: {}".format(num_keys))

    print("Reading features from:", args.feats_fn)
    if dataset == "buckeye":
        digit_list = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "oh"]
        initial_feats = np.load(args.feats_fn)
        initial_feats = dict(initial_feats)
        feats = {}
        for key in initial_feats:
            print(key.strip().split("_")[0])
            if key.strip().split("_")[0] not in digit_list:
                feats[key] = initial_feats[key]

        for key in feats:
            if key.strip().split("_") in digit_list: print(key)

    else:
        feats = np.load(args.feats_fn)
        feats = dict(feats)

    if args.normalize == "True":
        for utt in feats:
            N = feats[utt].shape[0]
            for i in range(N):
                feats[utt][i, :] = feats[utt][i, :]/np.linalg.norm(feats[utt][i, :])
    
    print("Calculating distances...")
    key_pair_file = open(args.key_pair_fn, 'w')
    for line in open(args.this_keys_list_fn, 'r'):

        distances = np.zeros(num_keys)
        keys = []
        key = line.strip()

        for i_pair, this_key in enumerate(open(args.all_keys_list_fn, 'r')):
            this_key = this_key.strip()

            distances[i_pair] = dist_func(
                np.array(feats[key], dtype=np.double), np.array(feats[this_key], dtype=np.double), True
                )
            keys.append(this_key)

        indices = np.argsort(distances)
        key_pair_file.write("{}".format(key))
        key_speaker = key.split("_")[1].split("-")[0]
        count = 0
        i = 0
        while count < args.num_pairs:
            cur_key = keys[indices[i]]
            cur_key_speaker = cur_key.split("_")[1].split("-")[0]
            i += 1

            if key_speaker != cur_key_speaker and key != cur_key:
                key_pair_file.write("\t{}".format(cur_key))
                count += 1
            if count == args.num_pairs: key_pair_file.write("\n")

    key_pair_file.close()
    print("End time: {}".format(datetime.datetime.now()))

if __name__ == "__main__":
    main()