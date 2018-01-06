""" Use ResNet to classify CIFAR images """

from __future__ import print_function

import os
import csv
import time
import shutil
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from model.resnet import resnet50
from model.vgg import vgg16_bn

from util.data.dataset import CifarDataset
from util.data.dataloader import CifarDataloader
from util.train import train
from util.test import test
from util.checkpoint import save_checkpoint, load_checkpoint


TRAIN_CSV_PATH = os.path.join('csv', 'train_labels.csv')
TRAIN_IMG_PATH = os.path.join('image', 'train')
TEST_CSV_PATH = os.path.join('csv', 'test_labels.csv')
TEST_IMG_PATH = os.path.join('image', 'test')

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

USE_GPU = True


def main():

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CifarDataset(TRAIN_CSV_PATH, TRAIN_IMG_PATH, transformations)
    train_loader = CifarDataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_dataset = CifarDataset(TEST_CSV_PATH, TEST_IMG_PATH, transformations)
    test_loader = CifarDataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = resnet50(pretrained=True, num_classes=10)
    criterion = nn.CrossEntropyLoss()

    if USE_GPU:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # load_checkpoint(os.path.join('checkpoint', 'last_checkpoint.pth.tar'), model, optimizer)

    for epoch in range(EPOCHS):
        train(train_loader, model, criterion, optimizer, epoch+1, USE_GPU)
        test(test_loader, model, USE_GPU)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join('checkpoint'))

if __name__ == "__main__":
    main()
