from __future__ import print_function
from __future__ import division

import os
import csv
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

def getLabelValue(label):
    if label == "airplane":
        return torch.LongTensor(np.array([0], dtype=np.int64))
    elif label == "automobile":
        return torch.LongTensor(np.array([1], dtype=np.int64))
    elif label == "bird":
        return torch.LongTensor(np.array([2], dtype=np.int64))
    elif label == "cat":
        return torch.LongTensor(np.array([3], dtype=np.int64))
    elif label == "deer":
        return torch.LongTensor(np.array([4], dtype=np.int64))
    elif label == "dog":
        return torch.LongTensor(np.array([5], dtype=np.int64))
    elif label == "frog":
        return torch.LongTensor(np.array([6], dtype=np.int64))
    elif label == "horse":
        return torch.LongTensor(np.array([7], dtype=np.int64))
    elif label == "ship":
        return torch.LongTensor(np.array([8], dtype=np.int64))
    elif label == "truck":
        return torch.LongTensor(np.array([9], dtype=np.int64))

class CifarDataset:

    def __init__(self, csv_path, img_path, transform=None):
        self.csv_path = csv_path
        self.img_path = img_path
        # obtain image filename and label list
        self.filenames = []
        self.labels = []
        csv_file = open(self.csv_path, 'r')
        for data in csv.DictReader(csv_file):
            self.filenames.append(data['filename'])
            self.labels.append(data['label'])
        csv_file.close()
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.filenames[index]))
        img = img.convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        label = getLabelValue(self.labels[index])
        return img, label

    def __len__(self):
        return len(self.labels)
