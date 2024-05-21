import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.Nets import *

def float2finitefield(net, args):
    for p in net.parameters():
        p.data = p.data * (2 ** args.field_size).int()
    return net

def finitefield2float(net, args):
    for p in net.parameters():
        p.data = p.data.to(torch.float) / (2 ** args.field_size)
    return net

