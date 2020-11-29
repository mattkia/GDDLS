import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

from source.data_loader import DatasetReader
from source.data_transforms import Rescale, ToTensor
##################################################
##################################################
dataset_name = 'HKU-IS'
image_size = 224
batch_size = 1
##################################################
##################################################
# Loading the Train and Test datasets
##################################################
print('[*] Loading the dataset...')
train_set = DatasetReader(dataset_name, train=True,
                          transform=transforms.Compose([Rescale((image_size, image_size)),
                                                        ToTensor()]))
test_set = DatasetReader(dataset_name, train=False,
                         transform=transforms.Compose([Rescale((image_size, image_size)),
                                                       ToTensor()]))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
##################################################
##################################################
# Saving the Train images
##################################################
print('[*] Saving the train images')
for i, sample in enumerate(train_loader):
    print('\t[*] instance ', i + 1, 'from  total ', len(train_set), 'instances')
    image, label = sample['image'], sample['label']

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalizer(image[0])

    np.save('np-datasets/' + dataset_name + '/Images/Train/' + str(i + 1) + '.npy', image.numpy())
    np.save('np-datasets/' + dataset_name + '/GT/Train/' + str(i + 1) + '.npy', label[0, 0].numpy())
##################################################
##################################################
# Saving the Test images
##################################################
print('[*] Saving the test images')
for i, sample in enumerate(test_loader):
    print('\t[*] instance ', i + 1, 'from total ', len(test_set), 'instances')
    image, label = sample['image'], sample['label']

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalizer(image[0])

    np.save('np-datasets/' + dataset_name + '/Images/Test/' + str(i + 1) + '.npy', image.numpy())
    np.save('np-datasets/' + dataset_name + '/GT/Test/' + str(i + 1) + '.npy', label[0, 0].numpy())
##################################################
##################################################
