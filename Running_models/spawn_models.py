#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This model spawns all the models we implemented in this paper. To describe the architectures we
# used, we use a shorthand of :
#                               => RNN(X) for a RNN layer of size X,
#                               => FC(X) for a fully-connected layer of size X and 
#                               => softmax(X) for a softmax layer of size X. 
#
# The speech classifier architecture is: 
#                               => 3 × RNN(400); 
#                               => FC(130) representation layer; 
#                               => softmax(5118). 
#The vision classifier architecture is: 
#                               => FC(512);
#                               => ReLU; 
#                               => FC(512); 
#                               => ReLU; 
#                               => FC(512); 
#                               => FC(130) representation layer; 
#                               => softmax(964). 
# The Siamese speech architecture is: 
#                               => 3 × RNN(400); 
#                               => FC(130), 
# The Siamese vision architecture is:
#                               => FC(512); 
#                               => ReLU; 
#                               => FC(512); 
#                               => ReLU; 
#                               => FC(512);
#                               => FC(130) representation layer. 
#The speech AE, CAE and AE-CAE has architectures of: 
#                               => 3 × RNN(400); 
#                               => FC(130) representation layer; 
#                               => FC(400); 
#                               => 3 × RNN(400); 
#                               => fully-connected layer of the same size as the input. 
#The vision AE, CAE and AE-CAE has architectures of: 
#                               => FC(512); 
#                               => ReLU; 
#                               => FC(512);
#                               => ReLU; FC(512); 
#                               => FC(130) representation layer; 
#                               => FC(512);
#                               => ReLU;
#                               =>  FC(512); 
#                               => ReLU; 
#                               => FC(512); 
#                               => ReLU; 
#                               => FC(784) (28 × 28 pixels).
#
# We use a batch size of 256 or 512 and varying epochs depending on which suits the model best. 
# We use a learning rate of 0.001 everywhere. 
#_________________________________________________________________________________________________

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

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEEDS = [1, 5, 10, 21, 42]

#_____________________________________________________________________________________________________________________________________
#
# Argument function
#
#_____________________________________________________________________________________________________________________________________

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_seeds", type=str, choices=["True", "False"], default="False")
    return parser.parse_args()

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def main():

    parameters = arguments()

    model_commands = [

        "--model_type ae --architecture rnn --data_type buckeye  --batch_size 512 --epochs 50 " 
        + "--features_type mfcc --train_tag gt --enc 400_400_400 --latent 130 ",

        "--model_type cae --architecture rnn --data_type buckeye --batch_size 512 --epochs 50 " 
        + "--features_type mfcc --train_tag gt --enc 400_400_400 --latent 130 ",

        "--model_type cae --architecture rnn --data_type buckeye --batch_size 512 --pretrain True --epochs 50 " 
        + "--features_type mfcc --train_tag gt --enc 400_400_400 --latent 130",


        "--model_type ae --architecture rnn --data_type TIDigits --batch_size 512 --epochs 50 " 
        + "--features_type mfcc --train_tag gt --enc 400_400_400 --latent 130 ",

        "--model_type cae --architecture rnn --data_type TIDigits --batch_size 512 --epochs 50 " 
        + "--features_type mfcc --train_tag gt --enc 400_400_400 --latent 130 ",

        "--model_type cae --architecture rnn --data_type TIDigits --batch_size 512 --pretrain True --epochs 50 " 
        + "--features_type mfcc --train_tag gt --enc 400_400_400 --latent 130 ",


        "--model_type ae --architecture fc --data_type omniglot --batch_size 256 --epochs 50 " 
        + "--enc 512_512_512 --latent 130",

        "--model_type cae --architecture fc --data_type omniglot --batch_size 64 --epochs 50 " 
        + "--enc 512_512_512 --latent 130",

        "--model_type cae --architecture fc --data_type omniglot --batch_size 256 --epochs 50 " 
        + "--enc 512_512_512 --latent 130 --pretrain True",


        "--model_type ae --architecture fc --data_type MNIST --batch_size 256 --epochs 50 " 
        + "--enc 512_512_512 --latent 130",

        "--model_type cae --architecture fc --data_type MNIST --batch_size 1024 --epochs 50 " 
        + "--enc 512_512_512 --latent 130",

        "--model_type cae --architecture fc --data_type MNIST --batch_size 1024 --epochs 50 " 
        + "--enc 512_512_512 --latent 130 --pretrain True",


        "--model_type classifier --architecture rnn --data_type buckeye  --batch_size 512 --epochs 50 " 
        + "--features_type mfcc --train_tag gt --enc 400_400_400 --latent 130 ",

        "--model_type classifier --architecture fc --data_type omniglot --batch_size 256 --epochs 50 " 
        + "--enc 512_512_512 --latent 130",



        "--model_type siamese --architecture rnn --data_type buckeye  --batch_size 256 --epochs 50 " 
        + "--features_type mfcc --train_tag gt --enc 400_400_400 --latent 130 ",

        "--model_type siamese --architecture fc --data_type omniglot --batch_size 256 --epochs 50 " 
        + "--enc 512_512_512 --latent 130",

    ]
    
    test_seeds = parameters.test_seeds == "True"

    for this_command in model_commands:
        if test_seeds:
            for rnd_seed in SEEDS:
                cmd = "./train_model.py " + this_command + " --rnd_seed {}".format(rnd_seed)
                print("-"*150)
                print("\nCommand: " + cmd)
                print("\n" + "-"*150)
                sys.stdout.flush()
                proc = subprocess.Popen(cmd, shell=True)
                proc.wait()

        else:
            cmd = "./train_model.py " + this_command
            print("-"*150)
            print("\nCommand: " + cmd)
            print("\n" + "-"*150)
            sys.stdout.flush()
            proc = subprocess.Popen(cmd, shell=True)
            proc.wait()
    
if __name__ == "__main__":
    main()