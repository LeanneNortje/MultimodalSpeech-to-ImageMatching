#!/usr/bin/env python

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
import random

sys.path.append(path.join("..", "..", ".."))
from paths import data_path
data_path = path.join("..", "..", "..", data_path)

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():

    train_fn = path.join(data_path, "TIDigits", "tidigits_fa", "train_word_align.ctm")
    test_fn = path.join(data_path, "TIDigits", "tidigits_fa", "test_word_align.ctm")

    out_fn_1 = path.join(data_path, "TIDigits", "tidigits_fa", "words.wrd")
    train_speakers = []
    train = []
    with open(train_fn, "r") as read_train:
        for line in read_train:
            utt, red, start, dur, label = line.strip().split()
            start = float(start)
            dur = float(dur)
            end = start + dur
            utterance = utt.replace("_", "-")
            train.append((utterance, start, end, label))
            train_speakers.append(utt.split("_")[0])

    test = []
    test_speakers = []
    with open(test_fn, "r") as read_test:
        for line in read_test:
            utt, red, start, dur, label = line.strip().split()
            start = float(start)
            dur = float(dur)
            end = start + dur
            utterance = utt.replace("_", "-")
            test.append((utterance, start, end, label))
            test_speakers.append(utt.split("_")[0])

    with open(out_fn_1, "w") as write_out_1:
        for utt, start, end, label in train:
            write_out_1.write(
                "{} {} {} {}\n".format(utt, start, end, label)
                )
        for utt, start, end, label in test:
            write_out_1.write(
                "{} {} {} {}\n".format(utt, start, end, label)
                )

    
    train_speaker_set = set(train_speakers)
    test_speaker_set = set(test_speakers)

    speaker_list = [i for i in train_speaker_set]
    train_num = int(len(train_speaker_set)*0.75)
    val_num = int(len(train_speaker_set) - train_num)
    
    indices = np.arange(0, len(train_speaker_set))
    np.random.shuffle(indices)

    train_speak_ind = []
    for i in range(train_num):
        train_speak_ind.append(indices[i])
    val_speak_ind = []
    for i in range(val_num): 
        val_speak_ind.append(indices[i+train_num])

    training_speakers = []
    for i in train_speak_ind:
        training_speakers.append(speaker_list[i])
    validation_speakers = []
    for i in val_speak_ind:
        validation_speakers.append(speaker_list[i])
    
    out_fn_2 = path.join(data_path, "TIDigits", "tidigits_fa", "train_speakers.list")
    with open(out_fn_2, "w") as write_train:
        for i in sorted(training_speakers):
            write_train.write(
                i + "\n"
                )       

    out_fn_3 = path.join(data_path, "TIDigits", "tidigits_fa", "val_speakers.list")
    with open(out_fn_3, "w") as write_val:
        for i in sorted(validation_speakers):
            write_val.write(
                i + "\n"
                ) 

    out_fn_4 = path.join(data_path, "TIDigits", "tidigits_fa", "test_speakers.list")
    with open(out_fn_4, "w") as write_test:
        for speaker in sorted(test_speaker_set):
            write_test.write(
                speaker + "\n"
                )       
            

if __name__ == "__main__":
    main()