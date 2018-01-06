from __future__ import division
from __future__ import print_function

import os
import numpy as np

import torch

from torch.autograd import Variable

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class Tester:

    def __init__(self, model, USE_GPU):
        self.model = model
        self.USE_GPU = USE_GPU
        self.checkpoint_epoch = 0

    def run(self, test_loader, checkpoint_path=None):
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        self.__test(test_loader, self.model, self.USE_GPU)

    def load_checkpoint(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.checkpoint_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))

    def __test(self, test_loader, model, USE_GPU):
        # switch to evaluate mode
        model.eval()

        class_correct = [0] * 10
        class_total = [0] * 10

        for b, (inputs, labels) in enumerate(test_loader):

            labels = torch.squeeze(labels, 1)

            if USE_GPU:
                inputs, labels = Variable(inputs, volatile=True).cuda(async=True), Variable(labels, volatile=True).cuda(async=True)
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            for j in range(predicted.data.size()[0]):

                predict = predicted.data[j]
                label = labels.data[j]

                if predict == label:
                    class_correct[label] += 1

                class_total[label] += 1

        print("EPOCH {}".format(self.checkpoint_epoch))
        for idx, clss_name in enumerate(CLASS_NAMES):
            print("Accuracy of {}: {:0.3}".format(clss_name, class_correct[idx]/class_total[idx]))
