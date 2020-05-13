#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
# Some fragment of code adapted from and credit given to: Herman Kamper
#_________________________________________________________________________________________________
#
# This script splits and runs a task on multiple CPU cores. 

from __future__ import division
from __future__ import print_function
import argparse
import subprocess
import sys
import re

#_____________________________________________________________________________________________________________________________________
#
# Argument function
#
#_____________________________________________________________________________________________________________________________________

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_job", type=int)
    parser.add_argument("end_job", type=int)
    parser.add_argument("log_fn", type=str)
    parser.add_argument("cmd", type=str)

    return parser.parse_args()

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________

def main():
    args = check_argv()

    process_id = -1
    for i in range(args.start_job, args.end_job+1):
        cur_cmd = re.sub("JOB", str(i), args.cmd)
        cur_log = re.sub("JOB", str(i), args.log_fn)

        process_id = subprocess.Popen(
            cur_cmd, shell=True, stderr=subprocess.STDOUT, 
            stdout=open(cur_log, "wb")
            ).pid
        print("Job {} with process ID {} created".format(i, process_id))


if __name__ == "__main__":
    main()