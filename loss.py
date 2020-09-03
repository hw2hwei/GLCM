import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def caption_loss(output, label, criterion):
    batch_size, length = output.size(0), output.size(1)

    loss = criterion(output.reshape(batch_size*length, -1), 
                      label.reshape(batch_size*length))

    return loss