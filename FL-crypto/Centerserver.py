from torchvision import datasets, transforms
# from opacus import PrivacyEngine

from utils.options import args_parser
from models.Nets import *
from Aggserver import Aggserver
from client import Client
from cryptoutils.BLS.agg_rev import keyGen


class Centerserver:
    def __init__(self, args):
        # hook = sy.TorchHook(torch)
        # args = args_parser()
        # args.device = torch.device(
        #     'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        # args.device = torch.device('cpu')
        self.args = args
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            self.img_size = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)[0][0].shape
        elif args.dataset == 'cifar':
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.img_size = datasets.CIFAR10('data/cifar', train=True, download=True, transform=trans_cifar)[0][0].shape
        mo = self.GenInitGlobalmodel()
        self.initGlobalmodel = mo.to(self.args.device)

    def online(self, n: 'the number of AggServers', m: 'the number of Clients'):

        # create AggServers
        Aggserver_ = []

        # create Clients
        Clients_ = []
        for i in range(m):
            Clients_.append(Client(id="Client{}".format(i), global_model=self.initGlobalmodel, args=self.args))

        for i in range(n):
            Aggserver_.append(Aggserver(id="Aggserver{}".format(i), global_model=self.initGlobalmodel, args=self.args))

        return Aggserver_, Clients_

    def GenInitGlobalmodel(self):
        if args_parser().model == 'cnn' and args_parser().dataset == 'mnist':
            print("CNN MNIST")
            self.net_global = CNNMnist(args=self.args).to(self.args.device)
        elif args_parser().model == 'cnn' and args_parser().dataset == 'cifar':
            self.net_global = CNNCifar(args=self.args).to(self.args.device)
        elif args_parser().model == 'mlp':
            len_in = 1
            for x in self.img_size:
                len_in *= x
            self.net_global = MLP(dim_in=len_in, dim_hidden=200, dim_out=self.args.num_classes).to(self.args.device)
        else:
            exit('Error: unrecognized model')
        self.net_global.train()

        return self.net_global

    def KeyGen(self,client_list,aggserver_list):
        pks = []
        pk2s = []
        for client in client_list:
            pk,pk2,sk = keyGen()
            client.recv_stack.append(sk)
            pks.append(pk)
            pk2s.append(pk2)
        for client in client_list:
            client.recv_stack.append(pks)
            client.recv_stack.append(pk2s)
