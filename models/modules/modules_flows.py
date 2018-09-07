#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 23:54:36 2018

@author: chemla
"""
import torch, pdb
import torch.nn as nn
from collections import OrderedDict

#%% Abst
class FlowItem(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super(FlowItem, self).__init__()
        self.dim = dim
        self.logdet = None
        
    def logdet_jacobian(self, z):
        return 0.
        

class Flow(nn.Module):
    def __init__(self, dim, n_flows, *args, **kwargs):
        super(Flow, self).__init__()
        self.dim = dim
        flows = OrderedDict()
        for i in range(n_flows):
            flows['flow_%d'%i] = self.create_flow(*args, **kwargs)
        self.flows = nn.Sequential(flows)

    def create_flow(self, *args, **kwargs):
        return FlowItem()        
    
    def forward(self, x):
        return self.flows(x)
    
    
#%% Planar flow
        
class PlanarFlowItem(FlowItem):
    def __repr__(self):
        return "PlanarFlowItem of dimension %d"%self.dim

    def logdet_jacobian(self, z):
        det = torch.baddbmm(self.bias.repeat(z.shape[0], 1).unsqueeze(2),
                           self.w.unsqueeze(0).repeat(z.shape[0], 1, 1).transpose(1,2), 
                           z.unsqueeze(2)) #+ b.repeat(z.shape[0]).
        det = torch.bmm(1 - torch.pow(torch.tanh(det),  2),
                        self.w.unsqueeze(0).repeat(z.shape[0], 1, 1).transpose(1,2))
        det = torch.baddbmm(torch.ones(z.shape[0], 1, 1),
                            det, self.u.unsqueeze(0).repeat(z.shape[0],1,1))
        det = det.squeeze().abs()
        return det

    def __init__(self, dim):
        super(PlanarFlowItem, self).__init__(dim)
        
        w = torch.nn.Parameter(torch.empty(dim, 1))
        self.register_parameter('w', w)
        nn.init.normal_(w)
        
        u = torch.nn.Parameter(torch.empty(dim, 1))
        self.register_parameter('u', u)
        nn.init.normal_(u)
        
        bias = torch.nn.Parameter(torch.zeros(1,1))
        self.register_parameter('bias', bias)
        nn.init.normal_(self.bias)

    def forward(self, z):
        res = torch.baddbmm(self.bias.repeat(z.shape[0], 1).unsqueeze(2),
                           self.w.unsqueeze(0).repeat(z.shape[0], 1, 1).transpose(1,2), 
                           z.unsqueeze(2)) #+ b.repeat(z.shape[0]).
        res = torch.baddbmm(z.unsqueeze(2),
                            self.u.unsqueeze(0).repeat(z.shape[0],1,1),
                            torch.tanh(res))
        res = res.squeeze()
        self.current_det = self.logdet_jacobian(z)
        return res
    
#    
    
class PlanarFlow(Flow):
    def create_flow(self, **kwargs):
        return PlanarFlowItem(self.dim)


#%%
     
if __name__ == "__main__":
    flow = PlanarFlow(20, 10)
    z = torch.zeros(100, 20).normal_()
    out = flow(z)        
    print(flow)