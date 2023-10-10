# Skin Canser Detection
> To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.


## Table of Contents
* [Data Reading/Data Understanding](#Data Reading/Data Understanding)
* [Dataset visualisation](#Dataset visualisation)
* [Model Building & training](#Model Building & training)
* [Data augmentation](#Data augmentation)
* [Model Building & training after data augmentation](#Model Building & training after data augmentation)
* [Class distribution](#Class distribution)
* [Handling class imbalances](#Handling class imbalances)
* [Model Building & training after handling class imbalance](#Model Building & training after handling class imbalance)

<!-- You can include any other section that is pertinent to your problem -->

## Data Reading/Data Understanding
- Using pathlib library we are reading images from the disk.
- Total 2357 images are provided for 9 different skin cancer diseases?
- Among 2357 images 2239 images are provided for model training purpose and remaining 118 are available for model evaluation?
- While creating dataset we are using image_dataset_from_directory utility from keras.preprocessing library.
- All the images are color images therefor images have 3 channels (RGB).
- While creating the training and validation datasets we are using 80% images for training i.e 1792 images and 20% images for validation i.e 447 images. We are also keeping the images size as 180 * 180 and batch size as 32. 
- Similarly we are creating test dataset from 118 images.

## Dataset visualisation
- There are total 9 types of skin cancer images are present in the dataset. The 9 classes are as follows
- 1 - actinic keratosis 
- 2 - basal cell carcinoma 
- 3 - dermatofibroma
- 4 - melanoma 
- 5 - nevus 
- 6 - pigmented benign keratosis 
- 7 - seborrheic keratosis
- 8 - squamous cell carcinoma 
- 9 - vascular lesion
- Datasets are Tensor image datasets with batch size 32 and 3 channels. The image batch is a tensor of the shape (32, 180, 180, 3) and label batch is a tensor of shape (32, 9).
- We have visualized one instance from all the 9 classes.

## Model Building & training
- As a initial model we are using 3 convolutional and pooling layers, one flatten layer, one dense and one output layer
- We are also using one layer for rescaling image values between 0-1
- In the first `convolution` layer (3*3) kernel will used to capture the 32 features of size (178, 178) from the dataset. Total 896 parametres will be trained. `Relu` optimization function will be used. After this convolution layer we are using maxpooling for aggrigating the information from the fetched features.
- In the second `convolution` layer (3*3) kernel will used to capture the 32 features of size (87, 87) from the features. Total 9248 parametres will be trained. `Relu` optimization function will be used. After this convolution layer we are using maxpooling (2*2) for aggrigating the information from the fetched features. 
- In the third `convolution` layer (3*3) kernel will used to capture the 32 features of size (41, 41) from the features. Total 9248 parametres will be trained. `Relu` optimization function will be used. After this convolution layer we are using maxpooling (2*2) for aggrigating the information from the fetched features. 
- After 3 continuous `convolution` and `maxpooling` layers we have used `flatten` layer.
- After `flatten` layer one `dense` layer with `relu` optimization function and one output layer with `softmax` optimization function is used to get 9 class classification.
- While compiling the model we are using `adam` optimizer, `categorical_crossentropy` loss and `accuracy` metrics. 
- While fitting or training the model we are using `20` epochs.
- After fitting the model we have got `80%` accuracy for traning dataset and `53%` accuracy for validation dataset.
- It is the sign of overfiting. 

## Data augmentation
- To handle the overfiting, we are using some data augmentation strategies.
- We have used randomflip and randomrotation strategy for flipping and rotating the images in the training dataset.
- This augmentation stategy is used for only training dataset only.

## Model Building & training after data augmentation
- In the above model.
- This augmentation stategy is used for training dataset only.
- We are also using batch normalization and dropout for handling overfiting.
- We have used 2 convolution layers with `32 (3*3) filters`, activation function `relu`. We also have 2 `batch normalization`, 1 `maxpooling` and one `dropout` layers.
- Again we have added 2 convolution layers with `64 (3*3) filters`, activation function `relu`. We also have 2 `batch normalization`, 1 `maxpooling` and one `dropout` layers.
- Again we have added 2 convolution layers with `128 (3*3) filters`, activation function `relu`. We also have 2 `batch normalization`, 1 `maxpooling` and one `dropout` layers.
- Then we have added `flatten` layer, one `dense` layer with `relu` optimization function and l2 regularization, one `dropout` layer, and one output layer with `softmax` optimization function is used to get 9 class classification.
- While compiling the model we are using `adam` optimizer, `categorical_crossentropy` loss and `accuracy` metrics. 
- While fitting or training the model we are using `20` epochs.
- After fitting the model we have got `61%` accuracy for traning dataset and `49%` accuracy for validation dataset.
- It is the sign of underfiting. 


## Class distribution
- We have checked the class wise number of samples present in the dataset.
- We found there is a class imbalance. Class `seborrheic keratosis` has the least no of samples as 77 samples. And class `pigmented benign keratosis`, `melanoma`, `basal cell carcinoma` dominates the dataset with 462, 438 and 376 samples respectively from total 2239 samples

## Handling class imbalances
- We have used Augmentor library to generate the new 500 images for each class.
- After generating new samples we have atleast sufficient class samples for each class to learn features and train model.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Model Building & training after handling class imbalance
- Now We are using same previous model created after data augmentation.
- While compiling the model we are using `adam` optimizer, `categorical_crossentropy` loss and `accuracy` metrics. 
- While fitting or training the model we are using `30` epochs.
- After fitting the model we have got `89%` accuracy for traning dataset and `86%` accuracy for validation dataset.
- Its a good improvement.
- We have evaluated the model on test dataset and have achieved 45% accuracy.  

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->
## Contact
Created by [@vaibhav-nk] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->