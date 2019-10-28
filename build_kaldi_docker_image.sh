#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

cd docker/
docker build -f Dockerfile.kaldi -t kaldi-tidigits-wav-conversion:latest .
