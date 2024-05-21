#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idx, total_num):
        self.dataset = dataset
        self.total_num = total_num
        self.idx = idx

        # 计算每份数据集的大小
        self.length = len(dataset) // total_num

        # 计算当前份数据集的起始索引和结束索引
        self.start_idx = self.idx * self.length
        self.end_idx = (self.idx + 1) * self.length

        # 处理最后一份数据集，保证所有样本都被分配到
        if self.idx == total_num - 1:
            self.end_idx = len(dataset)

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, item):
        index = self.start_idx + item
        return self.dataset[index]

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, total_num=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, total_num), batch_size=self.args.local_bs, shuffle=True)
        # self.ldr_train = DataLoader(dataset=dataset, batch_size=self.args.local_bs, shuffle=True)

        # self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        net.to(self.args.device)
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # print('images shape: ', images.shape)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels).to(self.args.device)
                optimizer.zero_grad()
                # print('gen loss')
                loss.backward()
                # print('do backward')
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

