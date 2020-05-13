#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
# Some fragment of code adapted from and credit given to: James Lyons, Python Speech Features, https://github.com/jameslyons/python_speech_features
#_________________________________________________________________________________________________
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
# Rounding function
#
#_____________________________________________________________________________________________________________________________________

def my_rounding(num):

    whole = float(int(num))

    if num - whole >= 0.5:
        return int(num) + 1
    else: 
        return int(num) 

#_____________________________________________________________________________________________________________________________________
#
# Speech processing blocks
#
#_____________________________________________________________________________________________________________________________________

def calculate_nfft(sampling_frequency, winlen):
    num_samples_in_frame = winlen * sampling_frequency
    nfft = 1
    while nfft < num_samples_in_frame:
        nfft *= 2
    return nfft

def preemphasis(wav_file, preemphasis_fact):
    signal = wav_file[1:] - preemphasis_fact * wav_file[:-1]
    return np.append(wav_file[0], signal)

def framing(sample_frequency, winlen, winstep, signal, winfunc):

    signal_length = len(signal)
    num_samples_in_frame = int(my_rounding(winlen * sample_frequency)) # frame_length
    num_samples_to_step = int(my_rounding(winstep * sample_frequency)) # frame_step

    if signal_length < num_samples_in_frame:
        num_frames = 1
    else:
        num_frames = int(np.ceil(float(np.abs(signal_length - num_samples_in_frame)) / num_samples_to_step)) + 1

    pad_length = int((num_frames - 1) * num_samples_to_step + num_samples_in_frame)
    zero_pad = np.zeros(pad_length - signal_length)
    padded_signal= np.append(signal, zero_pad)

    indices = np.tile(np.arange(0, num_samples_in_frame), (num_frames, 1)) + np.tile(np.arange(0, num_frames * num_samples_to_step, num_samples_to_step), (num_samples_in_frame, 1)).T
    frames = padded_signal[indices.astype(np.int32, copy=False)]
    win = np.tile(winfunc(num_samples_in_frame), (num_frames, 1))

    return frames * win

def power_spectrum(signal, nfft):

    if np.shape(signal)[1] > nfft:

        print(
            'Frame length (%d) is greater than nfft (%d), if nfft is not increased frames will be truncated.',
            np.shape(signal)[1], nfft)

    return np.square(np.absolute(np.fft.rfft(signal, nfft)))/nfft


def hztomel(f):

    return 2595 * np.log10(1 + f/700)

def meltohz(m):

    return 700*(10**(m/2595.0) - 1)

def fbank_melfilter(lowf, highf, nfilt, nfft, sampling_frequency):

    highfreq = highf or sampling_frequency/2
    assert highfreq <= sampling_frequency/2, "The high frequency is greater than the sampling frequency/2."
    
    lowmel = hztomel(lowf)
    highmel = hztomel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt+2)
    hzpoints = meltohz(melpoints)

    bin = np.floor((nfft+1)*(hzpoints)/sampling_frequency)  

    fbank = np.zeros([nfilt, int(nfft/2 + 1)])
    
    for m in range(0, nfilt):
        f_m_min_1 = int(bin[m])
        f_m = int(bin[m + 1])
        f_m_plus_1 = int(bin[m + 2]) 
        
        for k in range(f_m_min_1, f_m):
            fbank[m, k] =  (k - bin[m]) / (bin[m+1] - bin[m])

        for k in range(f_m, f_m_plus_1):
            fbank[m, k] =  (bin[m+2] - k) / (bin[m+2] - bin[m+1])

    return fbank
 

def filterbank(signal, sampling_frequency, preemphasis_fact, winlen, winstep, winfunc, lowf, highf, nfft, nfilt, return_energy = False, **kwargs):

    emphasized_signal = preemphasis(signal, preemphasis_fact)

    framed_and_windowed_signal = framing(sampling_frequency, winlen, winstep, emphasized_signal, winfunc)

    nfft = nfft if nfft is not None else calculate_nfft(sampling_frequency, winlen)
    power_signal = power_spectrum(framed_and_windowed_signal, nfft)

    energy = np.sum(power_signal,1)
    energy = np.where(energy == 0,np.finfo(float).eps,energy)

    fbank = fbank_melfilter(lowf, highf, nfilt, nfft, sampling_frequency)
    features = np.dot(power_signal, fbank.T)
    features = np.where(features == 0, np.finfo(float).eps, features)

    if return_energy: return energy, np.log(features) # dB
    else: return np.log(features)

def lifter(features, ceplifter):

    if ceplifter > 0:
        nframes,ncoeff = np.shape(features)
        n = np.arange(ncoeff)
        lift = 1 + (ceplifter/2.)*np.sin(np.pi*n/ceplifter)

        return lift*features
    else:
        return features


def mfcc(signal, sampling_frequency, preemphasis_fact, winlen, winstep, winfunc, lowf, highf, nfft, nfilt, numcep, ceplifter, append_energy, **kwargs):

    energy, feats = filterbank(
        signal, sampling_frequency, preemphasis_fact, winlen, winstep, winfunc, lowf, highf, nfft, nfilt, append_energy, **kwargs)

    mfccs = dct(feats, type=2, axis=1, norm='ortho')[:, :numcep]
    mfccs = lifter(mfccs, ceplifter)
    if append_energy: mfccs[:, 0] = np.log(energy)

    deltas = delta(mfccs, 2)
    delta_deltas = delta(deltas, 2)
    return np.hstack([mfccs, deltas, delta_deltas])

def delta(feats, N):

    if N < 1: print("Invalid N for delta calculation")

    denominator = 2 * sum([n**2 for n in range(1, N+1)])
    delta_feats = np.empty_like(feats)
    padded_feat = np.pad(feats, ((N, N), (0, 0)), mode='edge')
    
    for t in range(len(feats)):
        delta_feats[t] = np.dot(np.arange(-N, N+1), padded_feat[t: t+2*N+1])/denominator

    return delta_feats

#_____________________________________________________________________________________________________________________________________
#
# Speaker related functions
#
#_____________________________________________________________________________________________________________________________________

def get_speakers(feats):
    return set([key.split("_")[0] for key in feats])   

def speaker_mean_and_variance(feats, speakers):

    print("\nCalculating per speaker mean and variance:")
    speaker_feats = {}
    for key in sorted(feats):
        speaker = key.split("_")[0]
        if speaker not in speaker_feats:
            speaker_feats[speaker] = []
        speaker_feats[speaker].append(feats[key])

    mean_of_speaker = {}
    variance_of_speaker = {}
    for s in tqdm(speaker_feats):
        feats_of_speaker = np.vstack(speaker_feats[s])
        mean_of_speaker[s] = np.mean(feats_of_speaker, axis=0)
        variance_of_speaker[s] = np.std(feats_of_speaker, axis=0)

    return mean_of_speaker, variance_of_speaker

def speaker_mean_variance_normalization(feats, mean_of_speaker, variance_of_speaker):

    print("\nNormalizing for different speakers:")

    for key in tqdm(sorted(feats)):
        s = key.split("_")[0]
        feats[key] = ((feats[key] - mean_of_speaker[s]) / variance_of_speaker[s])

    return feats