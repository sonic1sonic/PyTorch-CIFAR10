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


class Trainer:

    def __init__(self, model, criterion, optimizer, USE_GPU):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.USE_GPU = USE_GPU
        self.start_epoch = 0

    def run(self, session, train_loader, checkpoint_path=None, run_epochs=1):
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        else:
            self.start_epoch = 0

        for epoch in range(run_epochs):
            self.__train(train_loader,
                         self.model,
                         self.criterion,
                         self.optimizer,
                         self.start_epoch+epoch+1,
                         self.USE_GPU)
            self.save_checkpoint({
                'epoch': self.start_epoch+epoch+1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, os.path.join('checkpoint'), session)
    
    def save_checkpoint(self, state, output_dir, session):
        if not os.path.exists(os.path.join(output_dir, session)):
            os.makedirs(os.path.join(output_dir, session))
        filepath = os.path.join(output_dir, session, 'epoch_{:03}.pth.tar'.format(state['epoch']))
        torch.save(state, filepath)
        shutil.copyfile(filepath, os.path.join(output_dir, session, 'last_checkpoint.pth.tar'))

    def load_checkpoint(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
    
    def __train(self, train_loader, model, criterion, optimizer, epoch, USE_GPU):
        # reset timer
        batch_time = 0.0
        # switch to train mode
        model.train()

        end = time.time()

        running_loss = 0.0

        for b, (inputs, labels) in enumerate(train_loader):

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

            if b%MSG_DISPLAY_FREQ == (MSG_DISPLAY_FREQ-1):
                print("[{}][{}/{}]\t Loss: {:0.5f}\t Batch time: {:0.3f}s"
                      .format(epoch, b+1, len(train_loader),
                              running_loss/MSG_DISPLAY_FREQ, batch_time/MSG_DISPLAY_FREQ))
                running_loss = 0.0
