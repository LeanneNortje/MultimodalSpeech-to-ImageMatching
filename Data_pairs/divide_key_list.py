#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
# Some fragment of code adapted from and credit given to: Herman Kamper
#_________________________________________________________________________________________________
#
# This script divides a long list of keys in a specified number N scripts.
#

from __future__ import division
from __future__ import print_function
import argparse
import sys
import numpy as np
import os
from os import path
import math

#_____________________________________________________________________________________________________________________________________
#
# Argument function
#
#_____________________________________________________________________________________________________________________________________


def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("keys_fn", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("num_files", type=int)
    return parser.parse_args()


#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________


def main():

    args = check_argv()

    if not path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    num_lines = sum([1 for line in open(args.keys_fn, 'r')])
    
    print("Number of keys: {}".format(num_lines))
    num_lines_per_file = int(math.ceil(float(num_lines)/args.num_files))
    print("Number of keys per file: {}".format(num_lines_per_file))

    basename, extension = path.splitext(path.split(args.keys_fn)[-1])

    i_file = 1
    line_count = 0
    cur_fn = path.join(args.output_dir, basename + "." + str(i_file) + extension)
    cur_file = open(cur_fn, "w")
    for line in open(args.keys_fn):

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