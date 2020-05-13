#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script installs all necessary GitHub repositories.  
#

git clone https://github.com/kamperh/speech_dtw.git /home/src/speech_dtw/
cd /home/src/speech_dtw/
make
make test

cd ../
git clone https://github.com/jameslyons/python_speech_features 
cd python_speech_features/
python setup.py develop