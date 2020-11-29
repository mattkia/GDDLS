import torch
import cv2

from skimage import transform


class ToTensor(object):
    def __init__(self, transpose=True):
        self.transpose = transpose

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        label = label.reshape(1, label.shape[0], label.shape[1])

        if self.transpose:
            image = image.transpose((2, 0, 1))

        return {'image': torch.tensor(image, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.float32)}


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        new_image = transform.resize(image, (new_h, new_w))
        new_label = transform.resize(label, (new_h, new_w))

        return {'image': new_image, 'label': new_label}



