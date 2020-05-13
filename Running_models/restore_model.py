#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script calculates restores speech or image models.  
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

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
            if lib["architecture"] == "rnn": speech_model_library.rnn_speech_siamese_model(lib)
        else:
            if lib["architecture"] == "rnn": speech_model_library.rnn_speech_model(lib)
            elif lib["architecture"] == "cnn": speech_model_library.cnn_speech_model(lib)
            elif lib["architecture"] == "fc": speech_model_library.fc_speech_model(lib)
    elif lib["training_on"] == "images": 
        if lib["model_type"] == "classifier":
            if lib["architecture"] == "fc": vision_model_library.fc_vision_classifier(lib)
        elif lib["model_type"] == "siamese":
            if lib["architecture"] == "fc": vision_model_library.fc_vision_siamese_model(lib)
        else:
            if lib["architecture"] == "cnn": vision_model_library.cnn_vision_model(lib)
            elif lib["architecture"] == "fc": vision_model_library.fc_vision_model(lib)

if __name__ == "__main__":
    main()
