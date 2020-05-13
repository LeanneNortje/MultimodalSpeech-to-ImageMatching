#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script contains various useful utility functions used in this repository.
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
import subprocess
import sys
from tqdm import tqdm
from scipy.fftpack import dct

#_____________________________________________________________________________________________________________________________________
#
# Directory related functions
#
#_____________________________________________________________________________________________________________________________________

def saving_path(directory, name):
	if not path.isdir(directory):
		os.makedirs(directory)

	return path.join(directory, name + ".ckpt")

def check_dir(directory):
    if not path.isdir(directory):
        os.makedirs(directory)



#_____________________________________________________________________________________________________________________________________
#
# Printing images
#
#_____________________________________________________________________________________________________________________________________

def print_image_double(name, image_one, image_two, image_number, size, fig_dir):
        
    image = np.empty((28 * image_number, 28 * 2))
    
    for j in range(image_number):
        image[j*28:(j+1)*28, 0*28:1*28] = image_one[j].reshape([28, 28])
        image[j*28:(j+1)*28, 1*28:2*28] = image_two[j].reshape([28, 28])
    plt.figure(figsize=(image_number*2*size, 2*size))
    plt.title(name)
    plt.imshow(image, origin="upper", cmap="gray")
    plt.axis("off")
    plt.show()
    plt.savefig(path.join(".", fig_dir + ".jpg"))

def print_image_double_breaks(name, image_one, image_two, image_number, size):
        
    image = np.empty((28, 28 * 2))
    
    for j in range(image_number):
        image[0*28:1*28, 0*28:1*28] = image_one[j].reshape([28, 28])
        image[0*28:1*28, 1*28:2*28] = image_two[j].reshape([28, 28])
        plt.figure(figsize=(image_number*2*size, 2*size))
        plt.title(name)
        plt.imshow(image, origin="upper", cmap="gray")
        plt.axis("off")
        plt.show()

def print_image_single(name, image_one, image_number, size, fig_dir):
    image = np.empty((28 * image_number, 28))

    
    for j in range(image_number):
        image[j*28:(j+1)*28, 0*28:1*28] = image_one[j].reshape([28, 28])
           
    plt.figure(figsize=(image_number*2*size, 2*size))
    plt.title(name)
    plt.imshow(image, origin="upper", cmap="gray")
    plt.axis("off")
    plt.show()
    plt.savefig(path.join(".", fig_dir + ".jpg"))

def print_image_single_breaks(name, image_one, image_number, size):
    image = np.empty((28, 28))

    
    for j in range(image_number):
        image[0*28:1*28, 0*28:1*28] = image_one[j].reshape([28, 28])
        plt.figure(figsize=(image_number*2*size, 2*size))
        plt.title(name)
        plt.imshow(image, origin="upper", cmap="gray")
        plt.axis("off")
        plt.show()

def print_image(name, input_image, size, fig_dir):
    image = np.empty((28, 28))

    image[0*28:1*28, 0*28:1*28] = input_image.reshape([28, 28])
    plt.figure(figsize=(2*size, 2*size))
    plt.title(name)
    plt.imshow(image, origin="upper", cmap="gray")
    plt.axis("off")
    plt.show()
    plt.savefig(path.join(".", fig_dir + ".jpg"))

#_____________________________________________________________________________________________________________________________________
#
# Printing speech spectograms
#
#_____________________________________________________________________________________________________________________________________

def spectogram(first, num_to_print, fig_dir):

    fig, array = plt.subplots(num_to_print, 1)
    for i, key in enumerate(first):
        if i == num_to_print: break
        array[i, 0].imshow(first[key])
        array[i, 0].axis("off")
    plt.savefig(path.join(".", fig_dir + "_spectogram.jpg"))


def double_spectogram(name, first, second, num_to_print, size, fig_dir):

    length = first.shape[-1]
    width = first.shape[-2]
    image = np.empty((num_to_print * length, width * 2))
    for j in range(num_to_print):
        image[j * length: (j + 1) * length, 0 * width: 1 * width] = first[j, :, :].reshape([length, width])
        image[j * length: (j + 1) * length, 1 * width: 2 * width] = second[j, :, :].reshape([length, width])
    plt.figure(figsize=(num_to_print*2*size, 2*size))
    plt.title(name)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    plt.savefig(path.join(".", fig_dir + "_spectogram.jpg"))