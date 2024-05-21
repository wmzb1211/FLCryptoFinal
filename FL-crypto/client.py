import random

from torchvision import datasets, transforms
# from opacus import PrivacyEngine

from models.Nets import *
from models.Update import LocalUpdate
from cryptoutils.BLS.agg_rev import sign, aggregate_verify, aggregate_sign, hashToPoint, BLS12_381_FQ2, subgroup_aggregate_verify
import time
import copy
class Client:


    def __init__(self, id: str, global_model, args):
        # hook = sy.TorchHook(torch)
        # args = args_parser()
        self.id = id
        # args.device = torch.device(
        #     'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        # args.device = torch.device('cpu')
        self.args = args
        # self.worker = sy.VirtualWorker(hook, id=self.id)
        self.global_model = global_model
        self.local_model = copy.deepcopy(self.global_model)
        self.send_stack = []
        self.recv_stack = []
        self.privKey = BLS12_381_FQ2()
        self.pubKeys = []
        self.pubKey2s = []
        self.hashs = []
        # self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.args.bs, shuffle=True)
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            self.dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
            self.dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)

        elif args.dataset == 'cifar':
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.dataset_train = datasets.CIFAR10('data/cifar', train=True, download=True, transform=trans_cifar)
            self.dataset_test = datasets.CIFAR10('data/cifar', train=False, download=True, transform=trans_cifar)
            # if args.iid:
            #     self.dict_users = cifar_iid(self.dataset_train, args.num_users)
            # else:
            #     exit('Error: only consider IID setting in CIFAR10')
        else:
            exit('Error: unrecognized dataset')
        self.img_size = self.dataset_train[0][0].shape


    def local_train(self):
        self.local_model = copy.deepcopy(self.global_model).to(self.args.device)
        self.global_model.train()
        model = self.local_model.to(self.args.device)
        # for name, param in model.state_dict().items():
        #     self.local_model.state_dict()[name].copy_(param.clone())

        # optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)


        local = LocalUpdate(args=self.args, dataset=self.dataset_train, idxs=int(self.id[-1]), total_num=self.args.num_users )
        # print('def localupdate done')
        w, loss = local.train(net=model)
        diff = dict()

        for name, param in w.items():
            diff[name] = param - self.global_model.state_dict()[name]

        return diff, loss

    # def Simple_SS(self, diff: dict) -> list:
    #     n = self.args.num_servers
    #     secrets = []
    #     sum = dict()
    #     for name, param in diff.items():
    #         sum[name] = torch.zeros_like(param)
    #     for i in range(n - 1):
    #         temp = dict()
    #         for name, param in diff.items():
    #             temp[name] = torch.rand_like(param) - 0.5
    #             sum[name] += temp[name]
    #         secrets.append(temp)
    #     six = dict()
    #     for name, param in diff.items():
    #         six[name] = param - sum[name]
    #
    #     secrets.append(six)
    #     return secrets
    def Simple_SS(self, diff: dict) -> list:
        n = self.args.num_servers
        secrets = []
        for i in range(n - 1):
            temp = dict()
            for name, param in diff.items():
                temp[name] = torch.rand_like(param) - 0.5
                diff[name] -= temp[name]
            secrets.append(temp)
        secrets.append(diff)
        return secrets


    # def send2aggserver(self, secrets: list, aggserver_list):
    #     for i in range(self.args.num_servers):
    #         temp = secrets[i] # dict()
    #         temp_model = self.global_model
    #         for name, param in temp_model.named_parameters():
    #             # temp_model.state_dict()[name] = temp[name]
    #             param.data = temp[name]
    #         # temp_model.load_state_dict(temp)
    #         aggserver_list[i].recv_stack.append(temp_model)
            # temp_model.send(aggserver_list[i].worker)
    def send2aggserver(self, secrets: list, aggserver_list, H, signs):
        for i in range(self.args.num_servers):
            temp = secrets[i]
            temp_model = dict()
            for name, param in self.global_model.named_parameters():
                temp_model[name] = temp[name]
            aggserver_list[i].recv_stack.append((temp_model,H,signs[i]))



    def aggmodel(self):
        # data_list = list(self.worker._objects.values())
        data_list = [data[0] for data in self.recv_stack]
        sum_result = data_list[0]
        paramnew = {}
        for name, param in sum_result.items():
            paramnew[name] = param
        for i in range(1, len(data_list)):
            for name, param in sum_result.items():
                paramnew[name] += data_list[i][name]

        for name, param in self.global_model.state_dict().items():
            paramnew[name] = (paramnew[name] + self.global_model.state_dict()[name])
        # self.global_model.load_state_dict(paramnew)
        for name, param in self.global_model.named_parameters():
            param.data = paramnew[name]
        self.local_model = copy.deepcopy(self.global_model).to(self.args.device)
        return self.global_model

    def downloadkeys(self):
        self.privKey = self.recv_stack[0]
        self.pubKeys = self.recv_stack[1]
        self.pubKey2s = self.recv_stack[2]
        self.recv_stack.clear()

    def diff2msg(self,diff: dict):
        return str(diff).encode('utf-8')

    def sign(self,secrets):
        H = BLS12_381_FQ2()
        signs = []
        i = 0
        for secret in secrets:
            # print('Sign for S{} ...'.format(i))
            # t1 = time.time()
            msg = self.diff2msg(secret)
            # t2 = time.time()
            h0 = hashToPoint(msg)
            # t3 = time.time()
            sgn = self.privKey * h0
            # t4 = time.time()
            # sgn = sign(msg,self.privKey)
            signs.append(sgn)
            H += h0
            # print('d2m',t2-t1)
            # print('htp',t3-t2)
            # print('cheng',t4-t3)
            # print('done.')
            i += 1
        return H,signs

    def aggregateverify(self):
        Hs = self.recv_stack[0][1]
        aggSigns = [data[2] for data in self.recv_stack]
        aggSign = aggregate_sign(aggSigns)
        alphas = []
        for _ in self.pubKey2s:
            alphas.append(random.randint(1,2**15))
        return aggregate_verify(Hs,aggSign,self.pubKeys,self.pubKey2s,alphas)

    def subgroupverify(self):
        Hs = self.recv_stack[0][1]
        aggSigns = [data[2] for data in self.recv_stack]
        aggSign = aggregate_sign(aggSigns)
        alphas = []
        for _ in self.pubKey2s:
            alphas.append(random.randint(1, 2 ** 15))
        return subgroup_aggregate_verify(Hs, aggSign, self.pubKeys, self.pubKey2s, alphas)