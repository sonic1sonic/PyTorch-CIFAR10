from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from torch.autograd import Variable

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def test(test_loader, model, USE_GPU):

    # switch to evaluate mode
    model.eval()

    class_correct = [0] * 10
    class_total = [0] * 10

    for i, (inputs, labels) in enumerate(test_loader):

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

    for idx, clss_name in enumerate(CLASS_NAMES):
        print("Accuracy of {}: {:0.3}".format(clss_name, class_correct[idx]/class_total[idx]))
