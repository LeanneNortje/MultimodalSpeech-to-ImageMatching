#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
#
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
import filecmp
import few_shot_learning_library

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
    """Check the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances_fn", type=str)
    parser.add_argument("--labels_fn", type=str)
    parser.add_argument("--binary_dists", type=str)
    parser.add_argument("--m_way", type=int)
    parser.add_argument("--k_shot", type=int)
    parser.add_argument("--test_fn", type=str)
    return parser.parse_args()


#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def main():

    args = check_argv()
    K = args.k_shot
    M = args.m_way

    if args.binary_dists == "True":
        print("Reading distances: {}".format(args.distances_fn))
        distances_vec = np.fromfile(args.distances_fn, dtype=np.float32)
    else:
        print("Reading distances: {}".format(args.distances_fn))
        distances_vec = np.fromfile(args.distances_fn, dtype=np.float32, sep="\n")

    if np.isnan(np.sum(distances_vec)):
        print("Warning: Distances contain nan") 

    num_queries = int(len(distances_vec)/(M*K))
    print(num_queries, M, K)
    num_episodes_check = int(sum([1 for line in open(args.labels_fn)])/(M*K))
    if num_queries != num_episodes_check: 
        print("Oops, file lengths don't match")

    label_grid = few_shot_learning_library.label_matches_grid_generation(args.labels_fn, num_queries, M, K)
    distance_grid = np.empty((num_queries, M*K))
    for n in range(len(distances_vec)):
        x = int(n/(M*K))
        y = int((n-(x*M*K))%(M*K))
        distance_grid[x, y] = distances_vec[n]

    if distance_grid.shape != label_grid.shape:
        print("Something went wrong when reading in the labels and distances")
    
    accuracy = 0

    closest_distances = np.argmin(distance_grid, axis=1)

    for i in range(len(closest_distances)):
        if label_grid[i, closest_distances[i]]: accuracy += 1

    print("Accuracy: {}".format(accuracy/len(closest_distances)))
    print("Accuracy: {:.2f}%".format(accuracy*100/len(closest_distances)))
    print("{} out of {} correct".format(accuracy, len(closest_distances)))

if __name__ == "__main__":
    main()