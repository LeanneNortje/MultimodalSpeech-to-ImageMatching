| Using `./train_model.py` to train models

To train a desired model, run:

```
cd Running_models/
./train_model.py --tag_1 <option_1> --tag_2 <option_2> .... --tag_N <option_N>
cd ../
```

where 1, 2, ..., N is any of the tags below with their appropriate options. 

| tag_n | option_n description |
| ----- | -------------------- |
| model_type | The type of model ("ae", "cae", "classifier", "siamese"). |
| architecture | The hidden layer structure to construct the architecture ("cnn", "rnn"). |
|final_model | ("True", "False") indicating whether this is a final model to be saved in a separate folder. |
|data_type | The dataset to train the model on ("buckeye", "TIDigits", "omniglot", "MNIST") {if you choose an image dataset, omniglot or MNIST, you can only choose the architecture as "cnn" and if you choose a speech dataset, buckeye or TIDigits, you can only choose the architecture as "rnn"}. |
|other_image_dataset | The image dataset to mix the the dataset specified in data_type with ("MNIST", "omniglot"). |
|other_speech_dataset | The speech dataset to mix the the dataset specified in data_type with ("buckeye", "TIDigits"). |
|features_type | The type of features to train on ("fbank", "mfcc", "None") {"fbank" and "mfcc" is applicable. |
|train_tag | The method in which spoken words for training are isolated ("gt", "None") {"gt" is applicable to speech features and "None" to images}. |
|max_frames | The maximum number of frames to limit the speech features to, not applicable to images. |
|mix_training_datasets | ("True", "False") indicating whether the datasets in data_type should be mixed with the other dataset. |
|train_model | ("True", "False") indicating whether the model should be trained {"False" will restore the model}. |
|use_best_model | ("True", "False") indicating whether the best model found with early stopping should be used  as the final trained model, "False" wil use the model produced at the last epoch. |
|test_model | ("True", "False") indicating whether the model should be tested. |
|activation | The name of the activation to use between layers ("relu", "sigmoid"). |
|batch_size | The size of the batches to dividing the dataset in each epoch. |
|n_buckets | Number of buckets to divide the dataset into according to the number of speech frames, not applicable to images
|margin | The hinge loss margin which is only applicable to the Siamese models. |
|sample_n_classes | The number of classes per Siamese batch to sample. |
|sample_k_examples | The number of examples to sample fo each of the sample_n_classes classes. |
|n_siamese_batches | The number of Siamese batches for each epoch. |
|rnn_type | The type of rnn cell to use in rnn layers. |
|epochs | The number of epochs to train the model for . |
|learning_rate | Any value to scale each parameter update. |
|keep_prob | The keep probability to use for each layer. |
|shuffle_batches_every_epoch | ("True", "False") indicating whether the data in each batch should be shuffled before being sent to the model. |
|divide_into_buckets | ("True", "False") indicating whether data should be divided into buckets according to the number of speech frames, not applicable to images. |
|one_shot_not_few_shot | ("True", "False") indicating to use one-shot of few-shot. |
|do_one_shot_test | ("True", "False") indicating whether the model should be tested on an unimodal one-shot classification task. |
|do_few_shot_test | ("True", "False") indicating whether the model should be tested on an unimodal few-shot classification task. |
|pair_type | "siamese", "classifier", "default") the type of distance metric used to generate pairs, the default is "cosine". |
|overwrite_pairs | ("True", "False") indicating whether ground truth labels should be used for data that are used as unlabelled. |
|pretrain | ("True", "False") indicating whether the model should be pretrained. |
|pretraining_model | The model type that the model should be pretrained as ("ae", "cae") {you can pretrain a model as itself and you can only pretrain a cae as an ae or vice versa}. |   
|pretraining_data | The dataset to pretrain the model on ("buckeye", "TIDigits", "omniglot", "MNIST") {if you choose an image dataset, omniglot or MNIST, you can only choose thearchitecture as "cnn" and if you choose a speech dataset, buckeye or TIDigits, you can only choose thearchitecture as "rnn"}. |
|pretraining_epochs | The number of epochs to pretrain the model for. |             
|other_pretraining_image_dataset | The image dataset to mix the the pretraining dataset specified in pretraining_data with ("MNIST", "omniglot"). |
|other_pretraining_speech_dataset | The speech dataset to mix the the pretraining dataset specified in pretraining_data with ("buckeye", "TIDigits"). |
|use_best_pretrained_model | ("True", "False") indicating whether the best pretrained model found with early stopping should be used to train the model from, "False" wil use the pretrained model produced at the last epoch. |
|M | The number of classes or concepts in the support set. |
|K | The number of examples of each class or concept in the support set. |
|Q | The number of queries in an episode. |
|one_shot_batches_to_use | The subset to use on a unimodal classification task for testing the model ("train", "validation", "test"). |
|one_shot_image_dataset | The image dataset to use on a unimodal classification task for testing a model ("MNIST", "omniglot"). |
|one_shot_speech_dataset | The speech dataset to use on a unimodal classification task for testing a model ("TIDigits", "buckeye"). |
|validation_image_dataset | The image dataset to use on a unimodal classification task for validation during training of a model ("MNIST", "omniglot"). |
|validation_speech_dataset | The speech dataset to use on a unimodal classification task for validation during training of a model ("TIDigits", "buckeye"). |
|test_on_one_shot_dataset | ("True", "False") indicating whether the unimodal classification task should be done on the specified one_shot_image_dataset or one_shot_speech_dataset. |
|validate_on_validation_dataset | ("True", "False") indicating whether the unimodal classification validation task should be done on the specified validation_image_dataset or validation_speech_dataset. |
|enc | The encoder layers given in a format where each layer dimension is divided by "_", i.e. 200_300_400 means an encoder with layer 3 layers of size 100, 200 and 300 in that precise order. |
|latent | The size of the latent or feature rpresentation. |
|latent_enc | Some encoder layers to encode the latent, given in a format where each layer dimension is divided by "_", i.e. 200_300_400 means an encoder with layer 3 layers of size 100, 200 and 300 in that precise order. |
|latent_func | The hidden layer structure to construct the latent enc-decoder to speech features and "None" to images}. |
|rnd_seed | The random seed used to initialize the random number generator. |
|