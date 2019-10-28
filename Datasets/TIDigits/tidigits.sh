#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script spawns the converion of the TIDigits sphere files to .wav files. 
#

print_help_and_exit=false

while getopts ":-:h" opt; do
	case $opt in 
		-)
			case $OPTARG in 
				tidigits|tidigits=*)

					arg=${OPTARG#tidigits}
					arg=${arg#*=}
					opt=${OPTARG%=${arg}}

					if [[ -z ${arg} ]]; then
						echo "Missing argument for --${opt}."
					else
						TIDIGITS_PATH=${arg}
					fi
					;;
			esac
	
	esac 
	
done

TIDIGITS_PATH=${TIDIGITS_PATH:-$PWD}
NUM_CORES=${NUM_CORES:-6}
DOCKER_NAME="kaldi-tidigits-wav-conversion"

docker run \
	-v "$(pwd)":/home \
	-e NUM_CORES=${NUM_CORES} \
	${DOCKER_NAME} \
	/bin/bash -c \
	"./convert_wavs.sh ; \
	chown -R $(id -u):$(id -g) /home"