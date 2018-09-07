import torch, pdb
from torch.autograd import Variable

import torch.distributions as dist

import sys
sys.path.append('../..')


# Kullback-Leibler divergences
def kld_gaussian_gaussian(gauss1, prior=[], logvar=False, size_average=False):
    mean1, std1 = gauss1
    if not logvar:
        std1 = torch.log(std1)
    if prior==[]:
        mean2, std2 = (torch.zeros_like(gauss1[0], device=mean1.device), torch.zeros_like(gauss1[1], device=std1.device))
    else:
        mean2, std2 = prior[0], prior[1]
        if not logvar:
            std2 = torch.log(std2)
    result = 0.5 * (std2 - std1 + torch.exp(std1-std2) + torch.pow(mean1-mean2,2)/torch.exp(std2) - 1)
    #result = torch.mean(torch.sum(result, 1))
    #result[:, 3:] = result[:, 3:]*20 # test horrible
    if size_average:
        result = torch.mean(result)
    else:
        result = torch.mean(torch.sum(result, 1))
    return result


def kld(dist1, dist2):
    if dist1==dist2==dist.Normal:
        return kld_gaussian_gaussian
    else:
        raise Exception("Don't find KLD module for distributions %s and %s"%(dist1, dist2))
