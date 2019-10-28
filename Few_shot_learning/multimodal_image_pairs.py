#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script identifies the closest spoken word in the support set to the query spoken word. Then
# sets up the lists used to calculate the distance between the closest spoken words' paired image
# and each image in the matching set. 
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
import generate_episodes
import math
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_fn", type=str)
    parser.add_argument("--distances_fn", type=str)
    parser.add_argument("--episode_order_fn", type=str)
    parser.add_argument("--support_set_fn", type=str)
    parser.add_argument("--labels_fn", type=str)
    parser.add_argument("--pair_labels_fn", type=str)
    parser.add_argument("--output_labels_fn", type=str)
    parser.add_argument("--keys_fn", type=str)
    parser.add_argument("--binary_dists", type=str)
    parser.add_argument("--m_way", type=int)
    parser.add_argument("--k_shot", type=int)
    parser.add_argument("--output_keys_fn", type=str)
    parser.add_argument("--num_files", type=int)
    parser.add_argument("--output_dir", type=str)
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
    
    distance_grid = np.empty((num_queries, M*K))

    for n in range(len(distances_vec)):
        x = int(n/(M*K))
        y = int((n-(x*M*K))%(M*K))
        distance_grid[x, y] = distances_vec[n]

    key_grid = few_shot_learning_library.key_grid_generation(args.keys_fn, num_queries, M, K)
    label_grid = few_shot_learning_library.key_grid_generation(args.pair_labels_fn, num_queries, M, K)

    if distance_grid.shape != key_grid.shape:
        print("Something went wrong when reading in the keys and distances")

    y_indexes = np.argmin(distance_grid, axis=1)

    print("Read in episode order form {}".format(args.episode_order_fn))
    episode_file = open(args.episode_order_fn, "r")
    episode_list = []
    for line in episode_file:
        episode = line.split()
        episode_list.extend(episode)
    episode_file.close()
    
    Q = int(num_queries/len(episode_list))

    print("Read in initial input labels from {}".format(args.labels_fn))
    labels_file = open(args.labels_fn, "r")
    labels_list = []
    for line in labels_file:
        label = line.strip().split()
        labels_list.extend(label)
    labels_file.close()

    test_label_file = open(args.test_fn, 'w')
    for i in range(len(y_indexes)):
        test_label_file.write("{} {}\n".format(labels_list[i], label_grid[i, y_indexes[i]].decode("utf-8")))
    test_label_file.close()

    print("Read in support set from {}".format(args.support_set_fn))
    support_set_file = open(args.support_set_fn, "r")
    support_set_list = []
    for i, line in enumerate(support_set_file):
        sp_key, im_key = line.strip().split()

        if i % (M*K) == 0: cur_support_set = {}
        cur_support_set[sp_key] = im_key
        if i - ((int(i/(M*K))) * (M*K)) == (M*K) - 1: support_set_list.append(cur_support_set)
    support_set_file.close()

    num_lines = 0
    episode_dict = generate_episodes.read_in_episodes(args.episodes_fn)
    output_keys_file = open(args.output_keys_fn, "w")
    output_labels_file = open(args.output_labels_fn, "w")
    for i in range(len(y_indexes)):

        episode = episode_list[int(i/Q)]
        
        cur_support_set = support_set_list[int(i/Q)]
        curr_episode = episode_dict[episode]
        matching_set = curr_episode["matching_set"]

        cur_sp_key = key_grid[i, y_indexes[i]].decode("utf-8")
        cur_im_key = cur_support_set[cur_sp_key]
        cur_input_label = labels_list[i]
        for j, match_key in enumerate(matching_set["keys"]):
            match_label = matching_set["labels"][j]
            output_keys_file.write("{} {}\n".format(cur_im_key, match_key))
            output_labels_file.write("{} {}\n".format(cur_input_label, match_label))
            num_lines += 1

    output_keys_file.close()
    output_labels_file.close()

    if not path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    print("Number of pairs: {}".format(num_lines))
    num_lines_per_file = int(math.ceil(float(num_lines)/args.num_files))
    print("Number of lines per file: {}".format(num_lines_per_file))

    basename, extension = path.splitext(path.split(args.output_keys_fn)[-1])

    i_file = 1
    line_count = 0
    cur_fn = path.join(args.output_dir, basename + "." + str(i_file) + extension)
    cur_file = open(cur_fn, "w")
    for line in open(args.output_keys_fn):

        cur_file.write(line)
        line_count += 1

        if line_count == num_lines_per_file:

            cur_file.close()

            if i_file == args.num_files:
                break

            i_file += 1
            line_count = 0
            cur_fn = path.join(args.output_dir, basename + "." + str(i_file) + extension)
            cur_file = open(cur_fn, "w")

    cur_file.close()

if __name__ == "__main__":
    main()