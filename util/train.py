from __future__ import division
from __future__ import print_function

import os
import time
import shutil

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from torch.autograd import Variable

MSG_DISPLAY_FREQ = 200


def train(train_loader, model, criterion, optimizer, epoch, USE_GPU=False):

    batch_time = 0.0

    # switch to train mode
    model.train()

    end = time.time()

    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):

        labels = torch.squeeze(labels, 1)

        if USE_GPU:
            inputs, labels = Variable(inputs).cuda(async=True), Variable(labels).cuda(async=True)
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        batch_time += time.time()-end
        end = time.time()

        if i % MSG_DISPLAY_FREQ == (MSG_DISPLAY_FREQ-1):
            print("[{}][{}/{}]\t Loss: {:0.5f}\t Batch time: {:0.3f}s".format(epoch, i+1, len(train_loader), running_loss/MSG_DISPLAY_FREQ, batch_time/MSG_DISPLAY_FREQ))
            running_loss = 0.0
