#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append("..")
# from ..utils.options import args_parser
# import syft as sy  # <-- NEW: import the PySyft library

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class ResBlk(nn.Module):
#     """
#     resnet block
#     """
#
#     def __init__(self, ch_in, ch_out, stride=1):
#         """
#
#         :param ch_in:
#         :param ch_out:
#         """
#         super(ResBlk, self).__init__()
#
#         self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
#         self.bn1 = nn.BatchNorm2d(ch_out)
#         self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(ch_out)
#
#         self.extra = nn.Sequential()
#
#         if ch_out != ch_in:
#             # [b, ch_in, h, w] => [b, ch_out, h, w]
#             self.extra = nn.Sequential(
#                 nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(ch_out)
#             )
#
#     def forward(self, x):
#         """
#
#         :param x: [b, ch, h, w]
#         :return:
#         """
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#
#         # short cut
#         # extra module:[b, ch_in, h, w] => [b, ch_out, h, w]
#         # element-wise add:
#         out = self.extra(x) + out
#         out = F.relu(out)
#
#         return out
#
#
# class ResNet18(nn.Module):
#     def __init__(self):
#         super(ResNet18, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=3, padding=0),
#             nn.BatchNorm2d(64)
#         )
#         # followed 4 blocks
#
#         # [b, 64, h, w] => [b, 128, h, w]
#         self.blk1 = ResBlk(64, 128, stride=2)
#
#         # [b, 128, h, w] => [b, 256, h, w]
#         self.blk2 = ResBlk(128, 256, stride=2)
#
#         # [b, 256, h, w] => [b, 512, h, w]
#         self.blk3 = ResBlk(256, 512, stride=2)
#
#         # [b, 512, h, w] => [b, 512, h, w]
#         self.blk4 = ResBlk(512, 512, stride=2)
#
#         self.outlayer = nn.Linear(512 * 1 * 1, 10)
#
#     def forward(self, x):
#         """
#
#         :param x:
#         :return:
#         """
#         # [b, 1, h, w] => [b, 64, h, w]
#         x = F.relu(self.conv1(x))
#
#         # [b, 64, h, w] => [b, 512, h, w]
#         x = self.blk1(x)
#         x = self.blk2(x)
#         x = self.blk3(x)
#         x = self.blk4(x)
#
#         # print(x.shape) # [b, 512, 1, 1]
#         # 意思就是不管之前的特征图尺寸为多少，只要设置为(1,1)，那么最终特征图大小都为(1,1)
#         # [b, 512, h, w] => [b, 512, 1, 1]
#         x = F.adaptive_avg_pool2d(x, [1, 1])
#         x = x.view(x.size(0), -1)
#         x = self.outlayer(x)
#
#         return x
#
#
# def main():
#     blk = ResBlk(1, 128, stride=4)
#     tmp = torch.randn(512, 1, 28, 28)
#     out = blk(tmp)
#     print('blk', out.shape)
#
#     model = ResNet18()
#     model_dict = model.state_dict()
#     secrets = Simple_SS(model_dict)
#     model_secrets = []
#     for i in range(len(secrets)):
#         model_secrets.append(secrets[i].load_state_dict(model_dict))
#
#     x = torch.randn(512, 1, 28, 28)
#     out = model(x)
#     print('resnet', out.shape)
#     print(model)
# def Simple_SS(diff: dict) -> list:
#     n = 4
#     secrets = []
#     for i in range(n - 1):
#
#
#
# if __name__ == '__main__':
#     main()

