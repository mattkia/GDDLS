import cv2
import os
import pathlib

import numpy as np
##################################################
##################################################
# loading the Train and Test data
##################################################
dataset_name = 'HKU-IS'

label_directory_train = str(pathlib.Path().absolute()) + '/datasets/' + dataset_name + '/GT/Train/'
label_directory_test = str(pathlib.Path().absolute()) + '/datasets/' + dataset_name + '/GT/Test/'

label_names_train = sorted([name for name in os.listdir(label_directory_train)
                            if os.path.isfile(os.path.join(label_directory_train, name))])
label_names_test = sorted([name for name in os.listdir(label_directory_test)
                           if os.path.isfile(os.path.join(label_directory_test, name))])
##################################################
##################################################
# Make the objects salient
##################################################
for name in label_names_train:
    image = cv2.imread('datasets/' + dataset_name + '/GT/Train/' + name, cv2.IMREAD_GRAYSCALE)
    image[np.where(image != 0)] = 255
    cv2.imwrite('datasets/' + dataset_name + '/GT/Train/' + name, image)
for name in label_names_test:
    image = cv2.imread('datasets/' + dataset_name + '/GT/Test/' + name, cv2.IMREAD_GRAYSCALE)
    image[np.where(image != 0)] = 255
    cv2.imwrite('datasets/' + dataset_name + '/GT/Test/' + name, image)
##################################################
##################################################
