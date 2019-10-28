#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

cd docker/
docker build -f Dockerfile.gpu -t gpu_docker_image .

# Installing GitHub repositories 

dirname=`dirname $(pwd)`

docker run --runtime=nvidia \
    -v "$dirname":/home --rm -it gpu_docker_image \
    /bin/bash -c "./docker/git_repos.sh; chown -R $(id -u):$(id -g) ./docker/git_repos.sh"