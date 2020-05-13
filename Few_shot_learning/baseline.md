## Baseline tests

For each of the scripts in this section, change the number of CPU cores you want to use. It is currently set to `6`. 

Go to the `Few_shot_learning` directory. 
```
cd Few_shot_learning/
```

# Unimodal one-shot image classification 

Firstly, run:
```
./unimodal_images_part_1.sh ../Few_shot_learning/Episode_files/M_10_K_1_Q_10_MNIST_test_episodes.txt ../Data_processing/MNIST/test.npz 10 1
```

Wait a while for the jobs to finish. `htop` is a nice tool to check when the jobs are done. 

```
./unimodal_images_part_2.sh ../Few_shot_learning/Episode_files/M_10_K_1_Q_10_MNIST_test_episodes.txt ../Data_processing/MNIST/test.npz 10 1
```

# Unimodal five-shot image classification 

Firstly, run:
```
./unimodal_images_part_1.sh ../Few_shot_learning/Episode_files/M_10_K_5_Q_10_MNIST_test_episodes.txt ../Data_processing/MNIST/test.npz 10 5
```

Wait a while for the jobs to finish. 

```
./unimodal_images_part_2.sh ../Few_shot_learning/Episode_files/M_10_K_5_Q_10_MNIST_test_episodes.txt ../Data_processing/MNIST/test.npz 10 5
```

# Unimodal one-shot speech classification 

Firstly, run:
```
./unimodal_speech_part_1.sh ../Few_shot_learning/Episode_files/M_11_K_1_Q_10_TIDigits_test_episodes.txt ../Data_processing/TIDigits/Subsets/Words/mfcc/gt_test_mfcc.npz 11 1
```

Wait a while for the jobs to finish. 

```
 ./unimodal_speech_part_2.sh ../Few_shot_learning/Episode_files/M_11_K_1_Q_10_TIDigits_test_episodes.txt ../Data_processing/TIDigits/Subsets/Words/mfcc/gt_test_mfcc.npz 11 1
```

# Unimodal five-shot speech classification 

Firstly, run:
```
./unimodal_speech_part_1.sh ../Few_shot_learning/Episode_files/M_11_K_5_Q_10_TIDigits_test_episodes.txt ../Data_processing/TIDigits/Subsets/Words/mfcc/gt_test_mfcc.npz 11 5
```

Wait a while for the jobs to finish. 

```
./unimodal_speech_part_2.sh ../Few_shot_learning/Episode_files/M_11_K_5_Q_10_TIDigits_test_episodes.txt ../Data_processing/TIDigits/Subsets/Words/mfcc/gt_test_mfcc.npz 11 5
```

# Multimodal one-shot speech-image matching 

Firstly, run:
```
./multimodal_part_1.sh ../Few_shot_learning/Episode_files/M_11_K_1_Q_10_TIDigits_test_MNIST_test_episodes.txt ../Data_processing/TIDigits/Subsets/Words/mfcc/gt_test_mfcc.npz 11 1 10
```

Wait a while for the jobs to finish. Secondly, run:

```
./multimodal_part_2.sh ../Few_shot_learning/Episode_files/M_11_K_1_Q_10_TIDigits_test_MNIST_test_episodes.txt ../Data_processing/MNIST/test.npz 11 1 10
```

Wait a while for the jobs to finish. Then, lastly, run:

```
./multimodal_part_3.sh ../Few_shot_learning/Episode_files/M_11_K_1_Q_10_TIDigits_test_MNIST_test_episodes.txt ../Data_processing/TIDigits/Subsets/Words/mfcc/gt_test_mfcc.npz 11 1 10
```

# Multimodal five-shot speech-image matching 

Firstly, run:
```
./multimodal_part_1.sh ../Few_shot_learning/Episode_files/M_11_K_5_Q_10_TIDigits_test_MNIST_test_episodes.txt ../Data_processing/TIDigits/Subsets/Words/mfcc/gt_test_mfcc.npz 11 5 10
```

Wait a while for the jobs to finish. Secondly, run:

```
./multimodal_part_2.sh ../Few_shot_learning/Episode_files/M_11_K_5_Q_10_TIDigits_test_MNIST_test_episodes.txt ../Data_processing/MNIST/test.npz 11 5 10
```

Wait a while for the jobs to finish. Then, lastly, run:

```
./multimodal_part_3.sh ../Few_shot_learning/Episode_files/M_11_K_5_Q_10_TIDigits_test_MNIST_test_episodes.txt ../Data_processing/MNIST/test.npz 11 5 10
```