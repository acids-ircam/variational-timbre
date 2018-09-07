import torch, pdb
from torch.nn import functional as F
from torch.autograd import Variable

from numpy import pi, log
import torch.distributions as dist

import sys
sys.path.append('../..')
from .. import distributions as cust





# adversarial loss
def get_adversarial_loss(d_fake, d_true, options={}):
#    print(torch.log(d_true), d_true)
    return -torch.mean(torch.log(d_true) + torch.log(1-d_fake))


# log-probabilities
def log_bernoulli(x, x_params, size_average=False):
    loss = F.binary_cross_entropy(x_params[0], x, size_average=size_average)
    if not size_average:
        loss = loss / x.size(0)
    return loss
    #return F.binary_cross_entropy(x_params[0], x, size_average = False)

def log_normal(x, x_params, logvar=False, clamp=True, size_average=False):
    if x_params == []:
        x_params = [torch.zeros_like(0, device=x.device), torch.zeros_like(0, device=x.device)]
    if len(x_params)<2:
        x_params.append(torch.full_like(x_params[0], 1e-3, device=x.device))
    mean, std = x_params
    if not logvar:
        std = std.log()
    # average probablities on batches
    #result = torch.mean(torch.sum(0.5*(std + (x-mean).pow(2).div(std.exp())+log(2*pi)), 1))
    loss = 0.5*(std + (x-mean).pow(2).div(std.exp())+log(2*pi))
    if size_average:
        loss = torch.mean(loss)
    else:
        loss = torch.mean(torch.sum(loss, 1))
    #result = F.mse_loss(x, x_params[0])
    return loss

def log_categorical(y, y_params, size_average=False):
    if y_params == []:
        y_params = y.clone().fill_(1/y.size(1))
    else:
        if issubclass(type(y_params), tuple):
            y_params = y_params[0]
    loss = torch.mean(torch.sum(y * y_params.log(), 1))
    if size_average:
        loss = torch.mean(loss)
    else:
        loss = torch.mean(torch.sum(loss, 1))

    return loss

def log_density(in_dist):
    if in_dist in [dist.Bernoulli, cust.AutoRegressive(dist.Bernoulli)]:
        return log_bernoulli
#    elif in_dist.dist_class==dist.normal.dist_class or in_dist.dist_class==cust.spectral.dist_class:
    elif in_dist in [dist.Normal, cust.Spectral, cust.AutoRegressive()]:
        return log_normal
    elif in_dist==dist.Categorical:
        return log_categorical
    else:
        raise Exception("Cannot find a criterion for distribution %s"%in_dist)
