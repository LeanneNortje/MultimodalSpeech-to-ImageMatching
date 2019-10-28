# Data pair generation

The files should be in the repository but these steps should reproduce the files. 

```
cd Data_pairs/
```

## MNIST cosine pairs

**Training pairs**

```
./get_image_pairs.sh ../Data_processing/MNIST/train.npz 5
```
Wait for jobs to finish then run:

```
./image_pair_list.sh ../Data_processing/MNIST/train.npz 5
```

**Validation pairs**

```
./get_image_pairs.sh ../Data_processing/MNIST/validation.npz 5
```
Wait for jobs to finish then run:

```
./image_pair_list.sh ../Data_processing/MNIST/validation.npz 5
```

**Testing pairs**

```
./get_image_pairs.sh ../Data_processing/MNIST/test.npz 5
```
Wait for jobs to finish then run:

```
./image_pair_list.sh ../Data_processing/MNIST/test.npz 5
```

## MNIST k-means pairs

**Training pairs**

```
./get_image_kmeans_pairs.sh ../Data_processing/MNIST/train.npz 5
```

**Validation pairs**

```
./get_image_kmeans_pairs.sh ../Data_processing/MNIST/validation.npz 5
```

**Testing pairs**

```
./get_image_kmeans_pairs.sh ../Data_processing/MNIST/test.npz 5
```

## Omniglot cosine pairs 

**Training pairs**

```
./get_image_pairs.sh ../Data_processing/omniglot/train.npz 5
```
Wait for jobs to finish then run:

```
./image_pair_list.sh ../Data_processing/omniglot/train.npz 5
```

**Testing pairs**

```
./get_image_pairs.sh ../Data_processing/omniglot/test.npz 5
```
Wait for jobs to finish then run:

```
./image_pair_list.sh ../Data_processing/omniglot/test.npz 5
```

## Omniglot k-means pairs 

**Training pairs**

```
./get_image_kmeans_pairs.sh ../Data_processing/omniglot/train.npz 5
```

**Testing pairs**

```
./get_image_kmeans_pairs.sh ../Data_processing/omniglot/test.npz 5
```

## TIDigts DTW pairs 

**Training pairs**

```
./get_speech_pairs.sh ../Data_processing/TIDigits/Subsets/Words/mfcc/gt_train_mfcc.npz 5
```
Wait for jobs to finish then run:

```
./speech_pair_list.sh ../Data_processing/TIDigits/Subsets/Words/mfcc/gt_train_mfcc.npz 5
```

**Validation pairs**

```
./get_speech_pairs.sh ../Data_processing/TIDigits/Subsets/Words/mfcc/gt_val_mfcc.npz 5
```
Wait for jobs to finish then run:

```
./speech_pair_list.sh ../Data_processing/TIDigits/Subsets/Words/mfcc/gt_val_mfcc.npz 5
```

**Testing pairs**

```
./get_speech_pairs.sh ../Data_processing/TIDigits/Subsets/Words/mfcc/gt_test_mfcc.npz 5
```
Wait for jobs to finish then run:

```
./speech_pair_list.sh ../Data_processing/TIDigits/Subsets/Words/mfcc/gt_test_mfcc.npz 5
```

## Buckeye DTW pairs

**Training pairs**

```
./get_speech_pairs.sh ../Data_processing/buckeye/Subsets/Words/mfcc/gt_train_mfcc.npz 5
```
Wait for jobs to finish then run:

```
./speech_pair_list.sh ../Data_processing/buckeye/Subsets/Words/mfcc/gt_train_mfcc.npz 5
```

**Validation pairs**

```
./get_speech_pairs.sh ../Data_processing/buckeye/Subsets/Words/mfcc/gt_val_mfcc.npz 5
```
Wait for jobs to finish then run:

```
./speech_pair_list.sh ../Data_processing/buckeye/Subsets/Words/mfcc/gt_val_mfcc.npz 5
```

**Testing pairs**

```
./get_speech_pairs.sh ../Data_processing/buckeye/Subsets/Words/mfcc/gt_test_mfcc.npz 5
```
Wait for jobs to finish then run:

```
./speech_pair_list.sh ../Data_processing/buckeye/Subsets/Words/mfcc/gt_test_mfcc.npz 5
```

```
cd ../
```