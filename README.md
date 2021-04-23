# Unsupervised vs transfer learning for multimodal one-shot matching of speech and images

## Overview

This repository includes the code used to implement the experiments in the paper: UNSUPERVISED VS TRANSFER LEARNING FOR MULTIMODAL ONE-SHOT MATCHING OF SPEECH AND IMAGES. Unsupervised approaches are compared to transfer learning from background labelled data. These two approaches are used to learn features used in a multimodal speech-image matching task. 

## Disclaimer

I provide no guarantees with this code, but I do believe the experiments in the above mentioned paper, can be reproduced with this code. Please notify me if you find any bugs or problems. 

## Installing `docker`

To install the latest `docker` for linux (ubuntu), follow these [steps](https://docs.docker.com/engine/install/ubuntu/).

## Installing `nvidia-docker`

Firstly make sure to follow the steps in [Installing docker](#Installing-docker). Then follow the steps [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

NOTE: If you intend on building the GPU-image in [Build docker images](#Build-docker-images), you have to install `nvidia-docker`. 

## Clone the repository 

To clone the repository run:

```
git clone https://github.com/LeanneNortje/multimodal_speech-image_matching.git
```

To get into the repository run:

```
cd multimodal_speech-image_matching/
```

## Build docker images

To install the `docker`-image and all the necessary repositories, do the following depending on whether you require the image for a CPU or GPU.

**CPU**

```
./build_cpu_docker_image.sh
```

**GPU**

```
./build_gpu_docker_image.sh
```

For more information regarding the base TensorFlow images used in these docker images, [visit](https://hub.docker.com/r/tensorflow/tensorflow/.)

To convert the TIDigit sphere files to .wav files, you have to build the kaldi docker image. Run:

```
./build_kaldi_docker_image.sh
```
## Running interactive shell in the `docker`-container

To run the `docker`-container each time, execute the following command depending on whether a CPU or GPU image were installed in [Build docker images](#Build-docker-images).

**CPU**

```
./run_cpu_docker_image.sh
```

**GPU**

```
./run_gpu_docker_image.sh
```

## Datasets


**Omniglot**
```
mkdir -p Datasets/Omniglot/
cd Datasets/Omniglot/
```

From [here](https://github.com/brendenlake/omniglot/tree/master/python) download the **images_background.zip** and **images_evaluation.zip** and copy it into the `Omniglot` folder. Follow these steps to extract the data. 


```
unzip images_background.zip
unzip images_evaluation.zip
```

Rename **images_background** to **train** and **images_evaluation** to **test** by running:

```
mv images_background train
mv images_evaluation test
cd ../../
```

**Buckeye**

Download the [Buckeye](https://buckeyecorpus.osu.edu/) corpus. Copy the folders s01-s40 into the `Datasets/buckeye/` folder. We use `english.wrd` to isolate each spoken word in te corpus. 

**TIDigits**

Download the [TIDigits](https://catalog.ldc.upenn.edu/LDC93S10) corpus. Copy the data and ensure the `tidigits` folder in the downloaded folder is in `Datasets/TIDigits/`. We divide the TIDigits corpus into subsets for training , validation  and testing according to the speakers listed in `train_speakers.list`, `val_speakers.list` and `test_speakers.list` respectively. We use `words.wrd` to isolate each spoken word in the corpus. 


**MNIST**

The MNIST corpus will automatically be downloaded during [Data processing](#Data-processing).

## Converting TIDigits sphere files to .wav files

When converting the TIDigits files, you should not currently be running in a docker image. After building the kaldi docker image in [Build docker images](#Build-docker-images), run:

```
cd Datasets/TIDigits/
./tidigits.sh
cd ../../
```

## Data processing

To process the data, follow the steps in [data_processing.md](Data_processing/data_processing.md).

## Episode generation

To generate all neccessary episodes with Q number of queries with a M-way K-shot support set, run:

```
cd Few_shot_learning/
./generate_all_episodes.sh
cd ../
```
 
## Data pairs

All the necessary pair files are already generated since it takes very long. The files are in the cloned repository. If neccessary the steps required to generate the pairs are in [data_pairs.md](Data_pairs/data_pairs.md) 

## Baseline tests

To run the baseline tests, follow the steps set out in [baseline.md](Few_shot_learning/baseline.md).

## Training models

To spawn all the models, except the classifier and Siamese models, we implemented in series, run:

```
cd Running_models/
./spawn_models.py --test_seeds <True, False>
cd ../
```

To spawn the classifier and Siamese models we implemented in series, and you haven't done so in [Data pairs](#Data-pairs) to generate new data pairs (and not use the provided), run:

```
cd Running_models/
./spawn_siamese_or_classifier_pairs.py --test_seeds <True, False>
cd ../
```

Instead of using `./spawn_models.py`, you can use `./train.py` to train a single model. To use the `./train.py` script follow these [instructions](Running_models/training_parameters.md).  

## Unimodal K-shot tests

To do the unimodal classification task for each model ran and saved in `Model_data`, run:

```
cd Model_results/
./spawn_unimodal_tests.py
```

This stores a log file in `/home/Model_results/Unimodal_results/` with all the unimodal classification accuracy results of a particular models' various trained instances. To get the overall models' accuracy mean and variance, run:

```
./get_mean_and_std_of_result.py
cd ../
```

## Multimodal K-shot tests

To do the multimodal speech-image matching task for a specific model pair:

```
cd Few_shot_learning/
./few_shot_learning.py --speech_log_fn <path to speech model log file in Model_data> \
--image_log_fn <path to image model log file in Model_data> --speech_data_fn <path to speech data> --image_data_fn <path to image data> --episode_fn <path to episode file>
cd ../
```

This stores a log file in `/home/Model_results/Multimodal_results/` with all the multimodal speech-image matching accuracy results of a particular speech-vision model pairs' various paired trained instances. Take note that the datasets given in `--speech_data_fn` and `--image_data_fn` should match to the datasets used to generate the episodes in `--episode_fn`. And `--k` in `./get_restult.py` should match to the `K`-shot episodes of `--episode_fn`. To do this for all the trained model pairs, simply run: 

```
cd Few_shot_learning/
./spawn_task.py
cd ../
```

To get the overall paired models' accuracy mean and variance, run:

```
cd Model_results/
./get_mean_and_std_of_result.py
cd ../
```
