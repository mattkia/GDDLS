import torch

import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

from source.data_loader import DatasetReader
from source.data_transforms import ToTensor
from source.models import vgg16, RiemannianManifoldVGGV2, SaliencyGuidedFilter
from source.utils import calculate_precision_recall, calculate_intersection_over_union
##################################################
##################################################
# Initializing the model parameters
##################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_names = ['CAMO-COCO', 'ECSSD', 'PASCAL', 'DUT-OMRON', 'HKU-IS']
dataset_name = dataset_names[4]
batch_size = 1
visual_mode = False
add_guided_filter = True
cross_validation = False
##################################################
##################################################
# Loading the Train and Test datasets
##################################################
print('[*] Loading the dataset ', dataset_name)
train_set = DatasetReader(dataset_name, train=True, mode='edited',
                          transform=transforms.Compose([ToTensor(transpose=False)]))

test_set = DatasetReader(dataset_name, train=False, mode='edited',
                         transform=transforms.Compose([ToTensor(transpose=False)]))


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
##################################################
# Initializing the model
##################################################
vgg = vgg16(pretrained=True)
model = RiemannianManifoldVGGV2().to(device)
saliency_map = SaliencyGuidedFilter(1, 1e-8)

if not cross_validation:
    if add_guided_filter:
        model.load_state_dict(torch.load('bce-saved-models/params_model_dgf_' + dataset_name + '.pt'))
        saliency_map.load_state_dict(torch.load('bce-saved-models/params_dgf_' + dataset_name + '.pt'))
    else:
        model.load_state_dict(torch.load('bce-saved-models/params_' + dataset_name + '.pt'))
else:
    if add_guided_filter:
        model.load_state_dict(torch.load('bce-saved-models/params_model_dgf_cv_' + dataset_name + '.pt'))
        saliency_map.load_state_dict(torch.load('bce-saved-models/params_dgf_cv_' + dataset_name + '.pt'))
    else:
        model.load_state_dict(torch.load('bce-saved-models/params_cv_' + dataset_name + '.pt'))
##################################################
##################################################
# Testing the model
##################################################
print('[*] Testing the model')
model.eval()
if add_guided_filter:
    saliency_map.eval()

if visual_mode:
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            image, label = sample['image'].to(device), sample['label'].to(device)

            out1 = vgg(image)
            out = model(out1)

            if add_guided_filter:
                out = saliency_map(image, out)

            # x = np.linspace(0, 223, 224)
            # y = np.linspace(0, 223, 224)
            # x, y = np.meshgrid(x, y)
            #
            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            # surf = ax.plot_surface(x, y, out[0, 0].numpy(), cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            # plt.show()

            img = torch.sigmoid(out)[0, 0].numpy()

            image = image[0].numpy().transpose(1, 2, 0)
            image = (image - np.min(image)) / (np.max(image) - np.min(image))

            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(label[0, 0].numpy())
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            if i == 9:
                exit()
else:
    with torch.no_grad():
        mae = 0.
        precision = 0.
        recall = 0.
        iou = 0.
        f_score = 0.
        for i, sample in enumerate(test_loader):
            image, label = sample['image'].to(device), sample['label'].to(device)

            out1 = vgg(image)
            out = model(out1)

            if add_guided_filter:
                out = saliency_map(image, out)

            heaviside = torch.sigmoid(out)[0, 0].numpy()

            mae += np.mean(np.abs(heaviside - label[0, 0].numpy()))

            p, r = calculate_precision_recall(heaviside, label[0, 0].numpy())
            precision += p
            recall += r

            iou += calculate_intersection_over_union(heaviside, label[0, 0].numpy())

            # f_score += calculate_omega_f_score(heaviside.numpy(), label[0, 0].numpy())

        mae = mae / len(test_loader)
        precision = precision / len(test_loader)
        recall = recall / len(test_loader)
        iou = iou / len(test_loader)

        print('MAE = ', mae)
        print('Precision = ', precision)
        print('Recall = ', recall)
        print('IOU = ', iou)
        print('Adaptive_F_beta = ', (1.3 * precision * recall) / (0.3 * precision + recall))
        # print('Omega-F_beta = ', f_score / len(test_loader))
##################################################
##################################################

