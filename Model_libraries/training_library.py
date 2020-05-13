#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script contains the training function to trainnn models.
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
from scipy.spatial.distance import pdist
import logging
import tensorflow as tf
import hashlib
import timeit
from tqdm import tqdm
import sys
import subprocess
import pickle

import model_legos_library

def training_model(training_parameters, input_placeholders, lib, train, epochs, patience, min_number_epochs, model, 
    val=None, val_tensor=None, restore_fn=None, save_model_fn=None, save_best_model_fn=None, pretraining=False):
    log = ""
    rec = {}
    rec["epoch_time"] = []
    rec["training_losses"] = []
    rec["validation_losses"] = []
    min_val_loss = np.inf
    not_save_counter = 0

    loss = training_parameters[0]
    optimization = training_parameters[1]
    train_flag = training_parameters[2]
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sesh = tf.Session(config=config)
    
    if pretraining:
        print("\n\tPretraining {} as {}...".format(lib["model_type"], model))
        log += "nPretraining {} as {}...\n".format(lib["model_type"], model)
    else:
        print("\n\tTraining {}...".format(model))
        log += "Training {}...\n".format(model)

    if restore_fn is not None:
        print("\n\tRestoring pretrained model {}...".format(restore_fn))
        log += "Restoring pretrained model {}...\n".format(restore_fn)
        saver.restore(sesh, restore_fn)
    else:
        init = tf.global_variables_initializer()
        sesh.run(init)

    epoch = 1
    while epoch <= epochs:
        start_time = timeit.default_timer()

        train_losses = []

        for values in train:
            feed_dict = model_legos_library.feeding_dict(input_placeholders, values, train_flag, True)
            train_opt, train_loss = sesh.run([optimization, loss], feed_dict)
            train_losses.append(train_loss)

        mean_train_loss = np.mean(train_losses)

        if val_tensor is not None:
            saver.save(sesh, save_model_fn)
            val_loss = val_tensor()
            mean_val_loss = val_loss[-1]
        
        else:
            val_losses = []
            for values in val:

                feed_dict = model_legos_library.feeding_dict(input_placeholders, values, train_flag, False)
                val_loss = sesh.run(loss, feed_dict)
                val_losses.append(val_loss)

            mean_val_loss = np.mean(val_losses)

        if save_best_model_fn is not None:
            save, saved_to_path, min_val_loss, not_save_counter = model_legos_library.saving_best_model(mean_val_loss, min_val_loss, sesh, save_best_model_fn, saver, not_save_counter) 
            if save == "Saved": 
                path_saved = saved_to_path 
        else:
            save = " - "
            not_save_counter = 0
            
        end_time = timeit.default_timer()   
        # print("\tEpoch {}: Training losses: {:.6f}, Validation losses: {:.6f}, Training Time: {:.6f} {}".format(epoch, mean_train_loss, mean_val_loss, end_time - start_time, save)) 
        print(f'\tEpoch {epoch:3}:\t{"Training losses: ":<15}{mean_train_loss:3.6f}\t{"Validation losses: ":<15}{mean_val_loss:3.6f}\t{"Training Time: ":<15}{end_time - start_time:3.6f}\t{save}')
        log += "Epoch {}: Training losses: {:.6f}, Validation losses: {:.6f}, Training Time: {:.6f} {}\n".format(epoch, mean_train_loss, mean_val_loss, end_time - start_time, save)
        rec["epoch_time"].append((epoch, end_time - start_time))
        rec["training_losses"].append((epoch, mean_train_loss))
        rec["validation_losses"].append((epoch, val_loss))

        if not_save_counter >= patience and epoch >= min_number_epochs: break
        if not_save_counter < patience and epoch == epochs: epochs += 10
        epoch += 1
        
    if epochs == 0: 
        mean_val_loss = 0.1
        save, saved_to_path, min_val_loss = model_legos_library.saving_best_model(mean_val_loss, min_val_loss, sesh, save_best_model_fn, saver)
        if save == "Saved": 
            path_saved = saved_to_path

    if save_best_model_fn is not None:  
        print("\n\tBest model saved to {}\n".format(path_saved))
        log += "\nBest model saved to {}\n".format(path_saved)

    if save_model_fn is not None:
        path_saved = model_legos_library.saving_model(sesh, save_model_fn, saver)
        print("\tLast model saved to {}\n".format(path_saved))
        log += "Last model saved to {}\n".format(path_saved)

    total_time = np.sum([i for j, i in rec["epoch_time"]])
    hours = int(int(total_time/60)/60)
    minutes = int((total_time - (hours*60*60))/60)
    seconds = (total_time - (hours*60*60) - (minutes*60))
    print("\tTotal training time: {} hr {} min {} sec".format(hours, minutes, seconds))
    log += "Total training time: {} hr {} min {} sec\n".format(hours, minutes, seconds)

    return rec, log
