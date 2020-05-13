#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script calculates the mean and variance unimodal or multimodal accuracy scores of multiple 
# identical models trained at different seeds. 
#

from datetime import datetime
from os import path
import argparse
import glob
import numpy as np
import os
from os import path
from scipy.io import wavfile
import matplotlib.pyplot as plt
import logging
import re

import subprocess
import sys
from tqdm import tqdm

sys.path.append("..")
from paths import model_lib_path
from paths import results_path
from paths import general_lib_path

sys.path.append(path.join("..", model_lib_path))
import model_setup_library

sys.path.append(path.join("..", general_lib_path))
import util_library

keywords = {
    "few_shot":"5-shot accuracy of ", 
    "one_shot": "One-shot accuracy of "
}

PRINT_LENGTH = model_setup_library.PRINT_LENGTH
#_____________________________________________________________________________________________________________________________________
#
# Argument function 
#
#_____________________________________________________________________________________________________________________________________

def result(log_fn, keyword, key):
    accuracies = []
    rnd_seeds = []

    for line in open(log_fn, 'r'):

        if re.search("Model name: ", line):
            model_name = line.strip().split(" ")[-1]

        if re.search(keyword, line):
            line_parts = line.strip().split(" ")
            keyword_parts = keyword.strip().split(" ")
            ind = np.where(np.asarray(line_parts) == keyword_parts[0])[0][0]
            acc = float(line_parts[ind+3])
            rnd_seed = line_parts[-1]

            if rnd_seed not in rnd_seeds:
                accuracies.append(acc)
                rnd_seeds.append(rnd_seed)

    if len(accuracies) == 0:
        print(f'Log file {log_fn} is empty.')
        return

    accuracies = np.asarray(accuracies)
    mean = np.mean(accuracies)
    std = np.std(accuracies)

    results_fn = os.path.dirname(log_fn)
    util_library.check_dir(results_fn)

    results_fn = path.join(results_fn, key + "_mean_and_std.txt")

    if os.path.isfile(results_fn) is False:
        print("\n" + "-"*PRINT_LENGTH)
        print(f'Calculating results for {log_fn}')
        print("-"*PRINT_LENGTH + "\n")
        print(f'\tMean: {mean*100:3.2f}%')
        print(f'\tStandard deviation: {std*100:3.2f}%\n')
        print(f'\tWriting: {results_fn}.\n')
        results_file = open(results_fn, 'w')
        results_file.write("Mean: {}\n".format(mean))
        results_file.write("Standard deviation: {}\n".format(std))
        results_file.write("\nMean: {}%\n".format(mean*100))
        results_file.write("Standard deviation: {}%\n".format(std*100))
        results_file.close()
    else:
        print(f'{results_fn} already exists.\n')
#_____________________________________________________________________________________________________________________________________
#
# Main 
#
#_____________________________________________________________________________________________________________________________________

def main():

	
    directories = os.walk("./Unimodal_results/")
    log_names = []
    for root, dirs, files in directories:
        for filename in files:
            if filename.split("_")[-1] == "log.txt" or filename.split("_")[-1] == "results.txt":
                log = path.join(root, filename)
                log_names.append(log)


    few_shot_keyword = "5-shot accuracy of "
    one_shot_keyword = "One-shot accuracy of "
    other_shot_keyword = "1-shot accuracy of " 
    few_shot_key = "5_shot"
    one_shot_key = "1_shot"

    for log_fn in log_names:
        result(log_fn, one_shot_keyword, one_shot_key)
        result(log_fn, few_shot_keyword, few_shot_key)


    directories = os.walk("./Multimodal_results/")
    log_names = []
    for root, dirs, files in directories:
    	for filename in files:
            if filename.split("_")[-1] == "results.txt":
                log = path.join(root, filename)
                log_names.append(log)


    few_shot_keyword = "5-shot accuracy of "
    one_shot_keyword = "1-shot accuracy of " 
    few_shot_key = "5_shot"
    one_shot_key = "1_shot"

    for log_fn in log_names:
        if int(log_fn.split("/")[-1].split("_")[0]) == 1: result(log_fn, one_shot_keyword, one_shot_key)
        else: result(log_fn, few_shot_keyword, few_shot_key)

    directories = os.walk("./Non_final_unimodal_results/")
    log_names = []
    for root, dirs, files in directories:
        for filename in files:
            if filename.split("_")[-1] == "log.txt":
                log = path.join(root, filename)
                log_names.append(log)


    few_shot_keyword = "5-shot accuracy of "
    one_shot_keyword = "One-shot accuracy of "
    zero_shot_keyword = "Zero-shot accuracy of "
    few_shot_key = "5_shot"
    one_shot_key = "1_shot"

    for log_fn in log_names:
        result(log_fn, one_shot_keyword, one_shot_key)
        result(log_fn, few_shot_keyword, few_shot_key)


    directories = os.walk("./Non_final_multimodal_results/")
    log_names = []
    for root, dirs, files in directories:
    	for filename in files:
            if filename.split("_")[-1] == "results.txt":
                log = path.join(root, filename)
                log_names.append(log)


    few_shot_keyword = "5-shot accuracy of "
    one_shot_keyword = "1-shot accuracy of " 
    few_shot_key = "5_shot"
    one_shot_key = "1_shot"

    for log_fn in log_names:
        if int(log_fn.split("/")[-1].split("_")[0]) == 1: result(log_fn, one_shot_keyword, one_shot_key)
        else: result(log_fn, few_shot_keyword, few_shot_key)
if __name__ == "__main__":
    main()
