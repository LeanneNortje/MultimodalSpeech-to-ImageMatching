#!/bin/bash

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2019
# Email: nortjeleanne@gmail.com
# Some fragment of code adapted from and credit given to: R. Eloff, H. A. Engelbrecht, H. Kamper, "Multimodal One-Shot Learning of Speech and Images,"  inin Proc. ICASSP, 2019
#_________________________________________________________________________________________________
#
# This script does the conversion from sphere files to .wav files. 
#

# ------------------------------------------------------------------------------
# Install pip and required python packages
# ------------------------------------------------------------------------------
#apt-get update -y \
#    && apt-get install -y --no-install-recommends python-pip \
#    && python -m pip install --upgrade setuptools wheel \
#    && python -m pip install \
#        numpy==1.15.0 \
#        pillow==5.3.0 \
#        matplotlib==2.2.3 \
#        nltk==3.3.0
python -c "import nltk; nltk.download('stopwords')"

# ------------------------------------------------------------------------------
# Prepare Kaldi tools for feature extraction:
# ------------------------------------------------------------------------------
mkdir -p /home/tmp/kaldi_stuff 
cp -rL /kaldi/egs/tidigits/s5/* /home/tmp/kaldi_stuff
cd /home/tmp/kaldi_stuff

echo '
    export train_cmd=run.pl
    export decode_cmd=run.pl
    export mkgraph_cmd=run.pl' > cmd.sh

# Set Kaldi tools path (from updated path.sh in kaldi/egs/wsj/s5)
echo '
    export KALDI_ROOT=/kaldi
    [ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
    export PATH=/home/tmp/kaldi_stuff/utils:$KALDI_ROOT/tools/openfst/bin:/home/tmp/kaldi_stuff:$PATH
    [ ! -f $KALDI_ROOT/tools/config/common_path.sh ] \
        && echo >&2 "The standard file ${KALDI_ROOT}/tools/config/common_path.sh is not present -> Exit!" \
        && exit 1
    . $KALDI_ROOT/tools/config/common_path.sh
    export LC_ALL=C' > path.sh

source cmd.sh
source path.sh  # many scripts depend on this file being present in cwd

rootdir=/home/tidigits
wav_dir=/home/tidigits_wavs
mkdir -p $wav_dir


find $rootdir/train -name '*.wav' > $rootdir/train/train.flist
n=`cat $rootdir/train/train.flist | wc -l`
[[ $n -eq 12549 ]] || echo "Unexpected number of training files $n versus 12549"

find $rootdir/test -name '*.wav' > $rootdir/test/test.flist
n=`cat $rootdir/test/test.flist | wc -l`
[[ $n -eq 12547 ]] || echo "Unexpected number of test files $n versus 12547"

sph2pipe=/kaldi/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Invalid command sph2pipe"
   exit 1
fi
# Prepare train and test data: convert to .wav format and extract meta-info
for set in train test; do

    cat $rootdir/$set/$set.flist \
        | perl -ane 'm|/home/tidigits/(\S+/)[1-9zo]+[ab]\.wav| || die "bad line $_"; print "/home/tidigits_wavs/$1\n"; ' \
        > $wav_dir/dirs.txt 

    while read -r line 
    do
        mkdir -p "$line"
    done < "$wav_dir/dirs.txt"

    cat $rootdir/$set/$set.flist \
        | perl -ane 'm|/home/tidigits/(\S+)| || die "bad line $_"; print "/home/tidigits_wavs/$1 $_"; ' \
        | sort \
        | awk '{print("'$sph2pipe' -f wav " $2 " " $1);}' \
        > $rootdir/$set/command.scp

    bash $rootdir/$set/command.scp
done
echo "home wav files done!"