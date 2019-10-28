#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

docker run --gpus all -u $(id -u):$(id -g) \
    -v "$(pwd)":/home --rm -it gpu_docker_image 