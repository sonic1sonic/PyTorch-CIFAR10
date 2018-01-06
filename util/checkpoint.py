from __future__ import print_function

import os
import torch
import shutil


def save_checkpoint(state, output_dir):
    if not os.path.exists(os.path.join(output_dir)):
        os.makedirs(os.path.join(output_dir))
    filepath = os.path.join(output_dir, 'epoch_{:03}.pth.tar'.format(state['epoch']))
    torch.save(state, filepath)
    shutil.copyfile(filepath, os.path.join(output_dir, 'last_checkpoint.pth.tar'))

def load_checkpoint(checkpoint_path, model, optimizer):
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
