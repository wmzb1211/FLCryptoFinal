# from opacus import PrivacyEngine
from models.Nets import *
from cryptoutils.BLS.agg_rev import aggregate_sign


class Aggserver:

    def __init__(self, id, global_model, args):
        # hook = sy.TorchHook(torch)
        self.id = id
        # args = args_parser()
        # args.device = torch.device(
        #     'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        # args.device = torch.device('cpu')
        self.args = args
        # self.worker = sy.VirtualWorker(hook, id=self.id)
        self.global_model = global_model
        self.send_stack = []
        self.recv_stack = []


    def aggreater(self):
        # data_list = list(self.worker._objects.values())
        data_list = [data[0] for data in self.recv_stack]
        sum_result = data_list[0]
        paramnew = {}
        for name, param in sum_result.items():
            paramnew[name] = sum_result[name]
        for i in range(1, len(data_list)):
            for name, param in sum_result.items():
                paramnew[name] += data_list[i][name]
        for name, param in sum_result.items():
            # sum_result.state_dict()[name] /= (len(data_list) / self.args.lr)
            paramnew[name] /= (len(data_list))
        # self.global_model.load_state_dict(paramnew)
        # for name, param in self.global_model.named_parameters():
        #     param.data = paramnew[name]

        return paramnew


    def send2clients(self, sum_result, client_list, Hs, aggSign):
        # send the sum_result to the clients
        for i in range(self.args.num_users):
            # sum_result.send(client_list[i].worker)
            client_list[i].recv_stack.append((sum_result,Hs,aggSign))


    def aggregatesign(self):
        Hs = [data[1] for data in self.recv_stack]
        sigs = [data[2] for data in self.recv_stack]
        aggSign = aggregate_sign(sigs)
        return Hs, aggSign
