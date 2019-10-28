#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
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

#_____________________________________________________________________________________________________________________________________
#
# Argument function 
#
#_____________________________________________________________________________________________________________________________________

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_fn", type=str)
    parser.add_argument("--one_not_few_shot", choices=["True", "False"], type=str, default="True")
    parser.add_argument("--k", type=int, default=-1)
    return parser.parse_args()

#_____________________________________________________________________________________________________________________________________
#
# Main 
#
#_____________________________________________________________________________________________________________________________________

def main():

	parameters = arguments()
	log_fn = parameters.log_fn
	if parameters.k != -1: 
		keyword = "{}-shot".format(parameters.k)
		key = "_{}_shot".format(parameters.k)
	elif parameters.one_not_few_shot == "True": 
		keyword = keywords["one_shot"]
		key = "_one_shot" 
	else: 
		keyword = keywords["few_shot"]
		key = "_few_shot"

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

	accuracies = np.asarray(accuracies)
	mean = np.mean(accuracies)
	std = np.std(accuracies)

	results_fn = os.path.dirname(parameters.log_fn)
	util_library.check_dir(results_fn)
	
	name = ("_").join([model_name, ("_").join(str(datetime.now()).split(" "))])
	results_fn = path.join(results_fn, name + key + "_results.txt")

	results_file = open(results_fn, 'w')
	results_file.write("Mean: {}\n".format(mean))
	results_file.write("Standard deviation: {}\n".format(std))
	results_file.write("\nMean: {}%\n".format(mean*100))
	results_file.write("Standard deviation: {}%\n".format(std*100))
	results_file.close()

if __name__ == "__main__":
    main()
