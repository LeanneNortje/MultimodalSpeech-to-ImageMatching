#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script lists all possible speech pairs in order to get the distances between each image and 
# every other image in the dataset. It also writes lists necessary for the mulltimodal matching task.
#

from __future__ import division
from __future__ import print_function
import argparse
import sys
import numpy as np
import os
from os import path
import math
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
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_fn", type=str)
    parser.add_argument("pairs_fn", type=str)
    parser.add_argument("label_pairs_fn", type=str)
    parser.add_argument("episode_list_fn", type=str)
    parser.add_argument("support_set_pairs_fn", type=str)
    parser.add_argument("num_files", type=int)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("M", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("test_labels", type=str)
    return parser.parse_args()


#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def main():
    args = check_argv()
    M = args.M
    K = args.K
    num_lines = 0

    episode_dict = generate_episodes.read_in_episodes(args.episodes_fn)

    key_file = open(args.pairs_fn, "w")
    label_file = open(args.label_pairs_fn, "w")
    support_set_file = open(args.support_set_pairs_fn, "w")
    episode_file = open(args.episode_list_fn, "w")
    test_label_file = open(args.test_labels, "w")
    episode_list = []
    temp = 0
    for episode in episode_dict:
        episode_file.write("{}\n".format(episode))
        curr_episode = episode_dict[episode]
        cur_query = curr_episode["query"]
        curr_support_set = curr_episode["support_set"]

        for i, query_key in enumerate(cur_query["keys"]):
            query_label = cur_query["labels"][i]
            label_file.write(query_label + "\n")
            for j, sp_key in enumerate(curr_support_set["speech_keys"]):
                cur_label = curr_support_set["labels"][int(j/K)]
                key_file.write(query_key + " " + sp_key + "\n")
                test_label_file.write(query_label + " " + cur_label + "\n")

                if i == 0:
                    paired_image_key = curr_support_set["image_keys"][j]
                    support_set_file.write(sp_key + " " + paired_image_key + "\n")
                num_lines += 1

        temp += 1
                
    key_file.close()
    label_file.close()
    support_set_file.close()
    episode_file.close()
    test_label_file.close()

    if not path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    print("Number of pairs: {}".format(num_lines))
    num_lines_per_file = int(math.ceil(float(num_lines)/args.num_files))
    print("Number of lines per file: {}".format(num_lines_per_file))

    basename, extension = path.splitext(path.split(args.pairs_fn)[-1])

    i_file = 1
    line_count = 0
    cur_fn = path.join(args.output_dir, basename + "." + str(i_file) + extension)
    cur_file = open(cur_fn, "w")
    for line in open(args.pairs_fn):

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
