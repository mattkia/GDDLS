"""
This file is a helper program to separate the Train/Test files.
Put your dataset in 'datasets/YOUR-DATASET-NAME'. 'YOUR-DATASET-NAME' folder must contain the folders
'YOUR-DATASET-NAME-Images' and 'YOUR-DATASET-NAME-GT'. Create the folders '/Images/Train', 'Images/Test'
'GT/Train' and 'GT/Test' in 'YOUR-DATASET-NAME' folder. The program will separate the train and test data
according to the specified train ratio
"""
import os
import shutil
import numpy as np
##################################################
##################################################
# Reading the dataset directory
##################################################
dataset_name = 'HKU-IS'
dataset_directory = 'datasets/' + dataset_name

images_directory = dataset_directory + '/' + dataset_name + '-Images'
labels_directory = dataset_directory + '/' + dataset_name + '-GT'

train_image_directory = dataset_directory + '/Images/Train/'
test_image_directory = dataset_directory + '/Images/Test/'
train_label_directory = dataset_directory + '/GT/Train/'
test_label_directory = dataset_directory + '/GT/Test/'

images_names = sorted(os.listdir(images_directory))
labels_names = sorted(os.listdir(labels_directory))
##################################################
##################################################
# Moving the train and test files
##################################################
train_ratio = 0.8
set_size = len(images_names)
train_size = int(set_size * train_ratio)
test_size = set_size - train_size
quotient = int(np.round(set_size / test_size))

for i, j in enumerate(zip(images_names, labels_names)):
    if i % quotient == 0:
        shutil.move(images_directory + '/' + j[0], test_image_directory + j[0])
        shutil.move(labels_directory + '/' + j[1], test_label_directory + j[1])
    else:
        shutil.move(images_directory + '/' + j[0], train_image_directory + j[0])
        shutil.move(labels_directory + '/' + j[1], train_label_directory + j[1])


print(train_size, len(os.listdir(train_image_directory)), len(os.listdir(train_label_directory)))
print(test_size, len(os.listdir(test_image_directory)), len(os.listdir(test_label_directory)))
##################################################
##################################################

