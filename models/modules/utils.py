# Initialization methods
import torch.nn as nn

DEFAULT_NNLIN = "ReLU"
DEFAULT_INIT = nn.init.xavier_normal_

def get_init(nn_lin):
    if nn_lin=="ReLU":
        return 'relu'
    elif nn_lin=="TanH":
        return 'tanh'
    elif nn_lin=="LeakyReLU":
        return 'leaky_relu'
    elif nn_lin=="conv1d":
        return "conv1d"
    elif nn_lin=="cov2d":
        return "conv2d"
    elif nn_lin=="conv3d":
        return "conv3d"
    elif nn_lin=="Sigmoid":
        return "sigmoid"
    else:
        return "linear"
    
def init_module(module, nn_lin=DEFAULT_NNLIN, method=DEFAULT_INIT):
    gain = nn.init.calculate_gain(get_init(nn_lin))
    if type(module)==nn.Sequential:
        for m in module:
            init_module(m, nn_lin=nn_lin, method=method)
    if type(module)==nn.Linear:
        method(module.weight.data, gain)
        module.bias.data.fill_(0.01)
