#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
# Some fragment of code adapted from and credit given to: Herman Kamper
#_________________________________________________________________________________________________
#
# This script lists all the data entries(keys) in a dataset. 
#

from __future__ import division
from __future__ import print_function
import argparse
import sys
import numpy as np

#_____________________________________________________________________________________________________________________________________
#
# Argument function
#
#_____________________________________________________________________________________________________________________________________

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("feats_fn", type=str)
    parser.add_argument("keys_fn", type=str)
    return parser.parse_args()


#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________


def main():
    args = check_argv()

    print(args.feats_fn)

    path_parts = args.feats_fn.strip().split("/")
    if len(path_parts) >= 5: dataset = path_parts[-5]
    else: dataset = path_parts[-2] 

    feats = np.load(args.feats_fn)

    print("Writing keys:", args.keys_fn)
    with open(args.keys_fn, "w") as f:
        for key in feats:
            if dataset == "buckeye" or dataset == "omniglot":
                digit_list = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "oh"]
                if key.strip().split("_")[0] not in digit_list:
                    f.write(key + "\n")

            else: f.write(key + "\n")

if __name__ == "__main__":
    main()
