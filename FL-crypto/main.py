import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import math
import numpy as np
from torchvision import datasets, transforms
import torch
import time
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
# import syft as sy
import random
from client import Client
from Aggserver import Aggserver
from Centerserver import Centerserver
from cryptoutils.BLS.agg_rev import pairing
from cryptoutils.BLS.curve.curve import G1, G2
from cryptoutils.BLS.agg_rev import aggregate_sign


def compare(w_avg, gmodel):
    Dict = {}
    for name, param in w_avg.items():
        temp = param - gmodel.state_dict()[name]
        Dict[name] = temp
    print(Dict)


def subgroup_verify_t(c, t):
    timefor = [0 for i in range(t)]
    verify_publictime_start = time.time()
    hashs = c.recv_stack[0][1]
    aggSigns = [data[2] for data in c.recv_stack]
    aggSign = aggregate_sign(aggSigns)
    alphas = []
    for _ in c.pubKeys:
        alphas.append(random.randint(1, 2 ** 15))
    pubKeys = c.pubKeys
    pubKey2s = c.pubKey2s
    verify_publictime_end = time.time()
    for i in range(t):
        timefor[i] += verify_publictime_end - verify_publictime_start
    verify_c1_start = time.time()
    p1_ = []
    for i in range(t):
        first_pairing_start = time.time()
        p1_.append(pairing(pubKeys[i], hashs[i] + alphas[i] * G2))
        first_pairing_end = time.time()
        timefor[i] += first_pairing_end - first_pairing_start
    for alpha, pubKey2 in zip(alphas, pubKey2s):
        aggSign += alpha * pubKey2
    p2 = pairing(G1, aggSign)
    for i in range(t, len(pubKeys)):
        verify_c_start =  time.time()
        p1_[i % t] = p1_[i % t] * pairing(pubKeys[i], hashs[i] + alphas[i] * G2)
        verify_c_end = time.time()
        timefor[i % t] += verify_c_end - verify_c_start
    p1 = p1_[0]
    for i in range(1, t):
        p1 = p1 * p1_[i]
    for i in range(t):
        print('time for {} is {}'.format(i, timefor[i]))
    if p1 == p2:
        print('verify success')
        return True
    else:
        print('verify fail')
        return False

def select_the_best_servers_and_subgroups(args):
    n = args.num_users
    m = args.num_servers
    if m < 0.848 * math.sqrt(n):
        t = m
    else:
        t = int(0.848 * math.sqrt(n))
    args.num_servers = t
    return args, t


if __name__ == '__main__':
    # hook = sy.TorchHook(torch)
    args = args_parser()
    if torch.cuda.is_available() and args.gpu != -1:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    print(args.device)

    # online
    time_detail_train = []
    time_detail_sign = []
    time_detail_verify = []
    CenterServer = Centerserver(args=args)
    Aggserver_list, Clients_list = CenterServer.online(n=args.num_servers, m=args.num_users)
    loss_total = []
    acc_total = []
    asr_total = []
    print('online done')
    print('there are {} clients and {} AggServers'.format(len(Clients_list), len(Aggserver_list)))
    # KeyGen
    CenterServer.KeyGen(Clients_list, Aggserver_list)
    print('Keys have been sent.')
    # downloadKeys
    for epoch in range(args.epochs):
        time_detail_train.append(0)
        time_detail_sign.append(0)
    for c in Clients_list:
        c.downloadkeys()
    print('Keys have been downloaded by clients.')
    for epoch in range(args.epochs):
        print('training')
        temploss = 0
        for c in Clients_list:
            c.recv_stack.clear()
        for Aggs in Aggserver_list:
            Aggs.recv_stack.clear()
        w_list = []
        for c in Clients_list:
            train_time = time.time()
            diff, loss = c.local_train()
            train_end = time.time()
            print('training time is: ', train_end - train_time)
            time_detail_train[epoch] += train_end - train_time
            print('local train done')
            temploss += loss
            secret = c.Simple_SS(diff)
            # sign
            starttime = time.time()
            H, signs = c.sign(secret)
            endtime = time.time()
            print('sign time is ', endtime - starttime)
            time_detail_sign[epoch] += endtime - starttime
            c.send2aggserver(secret, Aggserver_list, H, signs)
            print('send to Aggserver done')
        time_detail_sign[epoch] = time_detail_sign[epoch] / args.num_users
        time_detail_train[epoch] = time_detail_train[epoch] / args.num_users

        temploss = temploss / args.num_users
        loss_total.append(temploss)
        for Aggs in Aggserver_list:
            sum_result = Aggs.aggreater()
            # aggregate sign
            Hs, aggSign = Aggs.aggregatesign()
            Aggs.send2clients(sum_result, Clients_list, Hs, aggSign)
        c = Clients_list[0]
        globalModel = c.aggmodel()
        print('Verifying subgroup signature...')
        time1 = time.time()
        c1 = c
        c2 = Clients_list[1]
        args, t = select_the_best_servers_and_subgroups(args)
        subgroup_verify_t(c, t)


        for c in Clients_list:
            c.global_model = globalModel
            c.local_model = copy.deepcopy(globalModel)
        #
        print('epoch: ', epoch, ' loss: ', temploss)
        if epoch == args.epochs - 1:
            Final_ = Clients_list[0]
        elif epoch % 5 == 4:
            testmodel = Clients_list[0]
            testmodel.to(args.device)
            testmodel.eval()
            acctest, losstest = test_img(testmodel.global_model ,testmodel.dataset_test , args)
            print('acc: ', acctest)
    if not os.path.exists('./save/final'):
        os.mkdir('./save/final')
    plt.figure()
    plt.plot(range(len(loss_total)), loss_total)
    plt.ylabel('train_loss')
    plt.savefig('./save/final/loss_{}_{}_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs, args.num_users,
                                                              args.num_servers))
    plt.close()
    plt.figure()
    plt.plot(range(len(time_detail_train)), time_detail_train)
    plt.plot(range(len(time_detail_sign)), time_detail_sign)
    plt.plot(range(len(time_detail_verify)), time_detail_verify)
    plt.ylabel('time')
    plt.savefig('./save/final/time_{}_{}_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs, args.num_users,
                                                              args.num_servers))

    Final_.global_model.eval()
    acc_train, loss_train = test_img(Final_.global_model, Final_.dataset_train, args)
    acc_test, loss_test = test_img(Final_.global_model, Final_.dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))











