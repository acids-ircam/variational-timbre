#from pyro.nn import AutoRegressiveNN
import torch.nn as nn
from .utils import init_module
import torch.distributions as dist
from .. import distributions as cust





########################################################################
####        Gaussian layers

class GaussianLayer(nn.Module):
    '''Module that outputs parameters of a Gaussian distribution.'''
    def __init__(self, pinput, poutput):
        '''Args:
            pinput (dict): dimension of input
            poutput (dict): dimension of output
        '''
        super(GaussianLayer, self).__init__()
        self.input_dim = pinput['dim']; self.output_dim = poutput['dim']
        self.modules_list = nn.ModuleList()
        if issubclass(type(self.input_dim), list):
            input_dim_mean = self.input_dim[0]
            input_dim_var = self.input_dim[1]
        else:
            input_dim_mean = input_dim_var = self.input_dim
        mean_module = nn.Linear(input_dim_mean, self.output_dim)
        init_module(mean_module, 'Linear')
        self.modules_list.append(mean_module)
        var_module = nn.Sequential(nn.Linear(input_dim_var, self.output_dim), nn.Sigmoid())
        init_module(var_module, 'Sigmoid')
        self.modules_list.append(var_module)
        
    def forward(self, ins):
        '''Outputs parameters of a diabgonal Gaussian distribution.
        :param ins : input vector.
        :returns: (torch.Tensor, torch.Tensor)'''
        mu = self.modules_list[0](ins)
        logvar = self.modules_list[1](ins)
        return mu, logvar
    
class SpectralLayer(GaussianLayer):
    def __init__(self, pinput, poutput):
        super(SpectralLayer, self).__init__(pinput, poutput)
        self.modules_list[0] = nn.Sequential(self.modules_list[0], nn.Softplus())
        
    
class AutoRegressiveGaussianLayer(GaussianLayer):
    def __init__(self, pinput, poutput):
        super(AutoRegressiveGaussianLayer,  self).__init__(pinput, poutput)
        self.hidden_dim = pinput.get('hidden_dim') or 800
        autoregressive_module = nn.Sequential(nn.Linear(self.input_dim, self.output_dim),
                                          AutoRegressiveNN(self.output_dim, self.hidden_dim))
        init_module(autoregressive_module, 'linear')
        self.modules_list[0] = autoregressive_module
        
        
        
        
        
########################################################################
####        Bernoulli layers
      
class BernoulliLayer(nn.Module):
    '''Module that outputs parameters of a Bernoulli distribution.'''
    def __init__(self, pinput, poutput):
        super(BernoulliLayer, self).__init__()
        self.input_dim = pinput['dim']; self.output_dim = poutput['dim']
        self.modules_list = nn.Sequential(nn.Linear(self.input_dim, self.output_dim), nn.Sigmoid())
        init_module(self.modules_list, 'Sigmoid')
        
    def forward(self, ins):
        mu = self.modules_list(ins)
        return (mu,) 
    
class AutoRegressiveBernoulliLayer(nn.Module):
    '''Module that outputs parameters of a Bernoulli distribution.'''
    def __init__(self, pinput, poutput):
        super(AutoRegressiveBernoulliLayer, self).__init__()
        self.input_dim = pinput['dim']; self.output_dim = poutput['dim']
        self.hidden_dim = pinput.get('hidden_dim') or 800
        self.modules_list = nn.Sequential(nn.Linear(self.input_dim, self.output_dim), 
                                          AutoRegressiveNN(self.output_dim, self.hidden_dim),
                                          nn.Sigmoid())
        init_module(self.modules_list, 'Sigmoid')
        
    def forward(self, ins):
        mu = self.modules_list(ins)
        return (mu,) 





########################################################################
####        Categorical layers

class CategoricalLayer(nn.Module):
    '''Module that outputs parameters of a categorical distribution.'''
    def __init__(self, pinput, plabel):
        super(CategoricalLayer, self).__init__()
        self.input_dim = pinput['dim']; self.label_dim = plabel['dim']
        self.modules_list = nn.Sequential(nn.Linear(self.input_dim, self.label_dim), nn.Softmax())
        init_module(self.modules_list, 'Linear')
        
    def forward(self, ins):
        probs = self.modules_list(ins)
        return (probs,)



def get_module_from_density(distrib):
    if distrib == dist.Normal:
        return GaussianLayer
    elif distrib == dist.Bernoulli:
        return BernoulliLayer
    elif distrib == dist.Categorical:
        return CategoricalLayer
    elif distrib == cust.Spectral:
        return SpectralLayer
    elif distrib == cust.AutoRegressive(dist.normal):
        return AutoRegressiveGaussianLayer
    elif distrib == cust.AutoRegressive(dist.bernoulli):
        return AutoRegressiveBernoulliLayer
    else:
        raise TypeError('Unknown distribution type : %s'%distrib)
