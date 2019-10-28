#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script restores a model when given the path to its library. 
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
import tensorflow as tf

import subprocess
import sys
from tqdm import tqdm

sys.path.append("..")
from paths import model_lib_path
from paths import results_path
from paths import general_lib_path
from paths import general_lib_path

sys.path.append(path.join("..", model_lib_path))
import speech_model_library
import vision_model_library
import model_setup_library

sys.path.append(path.join("..", general_lib_path))
import util_library

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def main():


    lib = model_setup_library.restore_lib_from_arguments()

    lib["train_model"] = lib["pretrain"] = False

    if lib["training_on"] == "speech": 
        if lib["model_type"] == "classifier":
            if lib["architecture"] == "rnn": speech_model_library.rnn_speech_classifier(lib)
        elif lib["model_type"] == "siamese":
            if lib["architecture"] == "rnn": speech_model_library.rnn_speech_siamese(lib)
        else:
            if lib["architecture"] == "rnn": speech_model_library.rnn_speech_encdec(lib)

    elif lib["training_on"] == "images": 
        if lib["model_type"] == "classifier":
            if lib["architecture"] == "fc": vision_model_library.fc_vision_classifier(lib)
        elif lib["model_type"] == "siamese":
            if lib["architecture"] == "fc": vision_model_library.fc_vision_siamese(lib)
        else:
            if lib["architecture"] == "fc": vision_model_library.fc_vision_encdec(lib)

if __name__ == "__main__":
    main()
