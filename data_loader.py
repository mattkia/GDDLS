import torch
import cv2
import os
import pathlib

import numpy as np

from torch.utils.data import Dataset


class DatasetReader(Dataset):
    def __init__(self, dataset_name, transform=None, train=True, mode='original'):
        self.transform = transform
        self.dataset_name = dataset_name
        self.train = train
        self.mode = mode

        if train:
            if mode == 'original':
                data_directory = str(pathlib.Path().absolute()) + '/datasets/' + dataset_name + '/Images/Train/'
                label_directory = str(pathlib.Path().absolute()) + '/datasets/' + dataset_name + '/GT/Train/'
            elif mode == 'edited':
                data_directory = str(pathlib.Path().absolute()) + '/np-datasets/' + dataset_name + '/Images/Train/'
                label_directory = str(pathlib.Path().absolute()) + '/np-datasets/' + dataset_name + '/GT/Train/'
            else:
                raise Exception('[!] Bad data load mode...')
        else:
            if mode == 'original':
                data_directory = str(pathlib.Path().absolute()) + '/datasets/' + dataset_name + '/Images/Test/'
                label_directory = str(pathlib.Path().absolute()) + '/datasets/' + dataset_name + '/GT/Test/'
            elif mode == 'edited':
                data_directory = str(pathlib.Path().absolute()) + '/np-datasets/' + dataset_name + '/Images/Test/'
                label_directory = str(pathlib.Path().absolute()) + '/np-datasets/' + dataset_name + '/GT/Test/'
            else:
                raise Exception('[!] Bad data load mode...')

        # reading the images
        image_names = sorted([name for name in os.listdir(data_directory)
                              if os.path.isfile(os.path.join(data_directory, name))])
        label_names = sorted([name for name in os.listdir(label_directory)
                              if os.path.isfile(os.path.join(label_directory, name))])

        self.__image_names = image_names
        self.__label_names = label_names

    def __len__(self):
        return len(self.__image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            if self.mode == 'edited':
                image = np.load('np-datasets/' + self.dataset_name + '/Images/Train/' + self.__image_names[idx])
                label = np.load('np-datasets/' + self.dataset_name + '/GT/Train/' + self.__label_names[idx])
            elif self.mode == 'original':
                image = cv2.imread('datasets/' + self.dataset_name + '/Images/Train/' + self.__image_names[idx],
                                   cv2.IMREAD_COLOR)
                label = cv2.imread('datasets/' + self.dataset_name + '/GT/Train/' + self.__label_names[idx],
                                   cv2.IMREAD_GRAYSCALE)
            else:
                raise Exception('[!] Bad data load mode...')
        else:
            if self.mode == 'edited':
                image = np.load('np-datasets/' + self.dataset_name + '/Images/Test/' + self.__image_names[idx])
                label = np.load('np-datasets/' + self.dataset_name + '/GT/Test/' + self.__label_names[idx])
            elif self.mode == 'original':
                image = cv2.imread('datasets/' + self.dataset_name + '/Images/Test/' + self.__image_names[idx],
                                   cv2.IMREAD_COLOR)
                label = cv2.imread('datasets/' + self.dataset_name + '/GT/Test/' + self.__label_names[idx],
                                   cv2.IMREAD_GRAYSCALE)
            else:
                raise Exception('[!] Bad data load mode...')

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
