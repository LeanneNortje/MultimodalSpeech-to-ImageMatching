# Data processing

To process the data, follow the steps listed underneath each dataset. The TIDigit and Buckeye processing may take a while. 

## Omniglot

To downsample the images in the Omniglot dataset, run:

```
cd Data_processing/
./omniglot_subset_generation.py
```

## MNIST 

To download the MNIST dataset and generate a unique key for each image in the training, validation and testing datasets, run: 

```
./MNIST_subset_generation.py
```

## TIDgits and Buckeye

To parameterize all the speech audio in TIDigits and Buckeye, simply run:

```
./spawn_feature_extraction.py
``` 

This takes about 40 minutes. 

To remove digit classes from Buckeye and divide the corpus into training, validation and testing subsets with no class overlap, simply run:

```
./buckeye_subset_generation.py
cd ../
``` 