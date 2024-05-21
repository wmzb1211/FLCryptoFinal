from models.Nets import *
from models.Fed import FedAvg
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
my_model = MLP(dim_in=784, dim_hidden=200, dim_out=10).to(device)
for name, param in my_model.named_parameters():
    print(name, param.data)
for name, param in my_model.named_parameters():
    param.data = torch.ones_like(param.data)
for name, param in my_model.named_parameters():
    print(name, param.data)
# Modelparam = []
# for i in range(10):
#     param_new = {}
#     for name, param in my_model.named_parameters():
#         param_new[name] = torch.ones_like(param.data)
#     Modelparam.append(param_new)
#
# fedmodel = FedAvg(Modelparam)
#
# print(fedmodel)
#
#
# def simple_SS(modelparam: dict):
#     secret = []
#     Sum = {}
#     for name, param in modelparam.items():
#         Sum[name] = torch.zeros_like(param)
#     for i in range(5 - 1):
#         new = {}
#         for name, param in modelparam.items():
#             new[name] = torch.rand_like(param)
#             Sum[name] += new[name]
#         secret.append(new)
#     for name, param in modelparam.items():
#         modelparam[name] = modelparam[name] - Sum[name]
#     secret.append(modelparam)
#     return secret
#
# def aggmodel(secret: list):
#     sum_result = secret[0]
#     paramnew = {}
#     for name, param in sum_result.items():
#         paramnew[name] = sum_result[name]
#     for i in range(1, len(secret)):
#         for name, param in sum_result.items():
#             paramnew[name] += secret[i][name]
#     return paramnew
#
# secrets = []
# for i in range(10):
#     secrets_temp = simple_SS(Modelparam[i])
#     secrets.append(secrets_temp)
#
# paramtemp = []
#
# for i in range(5):
#     temp = []
#     for j in range(10):
#         temp.append(secrets[j][i])
#     paramtemp.append(aggmodel(temp))
#
# paramtemp = aggmodel(paramtemp)
#
# for name, param in paramtemp.items():
#     print(param / 10 - fedmodel[name])

