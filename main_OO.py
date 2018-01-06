""" Use ResNet to classify CIFAR images """

from __future__ import print_function

import os
import csv
import time
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
from util.trainer import Trainer
from util.tester import Tester


TRAIN_CSV_PATH = os.path.join('csv', 'train_labels.csv')
TRAIN_IMG_PATH = os.path.join('image', 'train')
TEST_CSV_PATH = os.path.join('csv', 'test_labels.csv')
TEST_IMG_PATH = os.path.join('image', 'test')

EPOCH = 20
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

    session_name = 'resnet50_cifar10'
    trainer = Trainer(model, criterion, optimizer, USE_GPU)
    tester = Tester(model, USE_GPU)

    trainer.run(session_name, train_loader, run_epochs=EPOCH)
    tester.run(test_loader,
               os.path.join('checkpoint', session_name, 'last_checkpoint.pth.tar'))

    print("Finished Training")

if __name__ == "__main__":
    main()
