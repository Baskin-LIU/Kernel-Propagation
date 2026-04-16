import numpy as np
import torch

class rho():
    def __init__(self, n_neuron, *args, **kwargs):
        self.defaultd=None
        return

    def __call__(self, x):
        return 

    def derivation(self, x):
        return

    def reset(self, ):
        self.d=self.defaultd


class linear(rho):
    def __init__(self, n_neuron, *args, **kwargs):
        self.d = torch.ones(n_neuron)
        self.defaultd = torch.ones(n_neuron)

    def __call__(self, x):
        self.d = torch.ones_like(x)
        return x

    def derivation(self, x):
        return torch.ones_like(x)


class relu(rho):
    def __init__(self, n_neuron, *args, **kwargs):
        self.d = torch.ones(n_neuron)
        self.defaultd = torch.ones(n_neuron)

    def __call__(self, x):
        self.d = (x>0).to(torch.float)
        return torch.relu(x)

    def derivation(self, x):
        return (x>0).to(torch.float)


class sigmoid(rho):
    def __init__(self, n_neuron, scale=1.6, *args, **kwargs):
        self.d = torch.zeros(n_neuron)
        self.defaultd = torch.zeros(n_neuron)
        self.scale = scale

    def __call__(self, x):
        a = torch.sigmoid(self.scale*x)
        self.d = self.scale*a*(1.-a)
        return a

    def derivation(self, x):
        a = torch.sigmoid(self.scale*x)
        return self.scale*a*(1-a)
        

class tanh(rho):
    def __init__(self, n_neuron, scale=1.6, *args, **kwargs):
        self.d = torch.zeros(n_neuron)
        self.defaultd = torch.zeros(n_neuron)
        self.scale = scale

    def __call__(self, x):
        a = torch.tanh(self.scale*x)
        self.d = (1.0 - a**2)*self.scale
        return a

    def derivation(self, x):
        a = torch.tanh(self.scale*x)
        return self.scale*(1.0 - a**2) #TODO match up version


class softplus(rho):
    def __init__(self, n_neuron, *args, **kwargs):
        self.d = torch.zeros(n_neuron)
        self.defaultd = torch.zeros(n_neuron)

    def __call__(self, x):
        expx = torch.exp(x)
        a = torch.log(1+expx)
        self.d = expx/(1+expx)
        return a

    def derivation(self, x):
        expx=torch.exp(x)
        return expx/(1+expx)


class wta(rho):
    def __init__(self, n_neuron, threshold=1., *args, **kwargs):
        self.d = torch.zeros(n_neuron)
        self.threshold = threshold
        self.defaultd = torch.zeros(n_neuron)

    def __call__(self, x):
        self.d *= 0
        if (x>self.threshold).any():
            winner = torch.argmax(x)
            self.d[winner] = 1
        a = x * self.d
        return a

    def derivation(self, x):
        winner = torch.argmax(x)
        d = torch.zeros_like(x)
        d[winner] = 1
        return d


class spiking(rho):
    def __init__(self, n_neuron, threshold, dt, *args, **kwargs):
        self.d = torch.zeros(n_neuron)
        self.defaultd = torch.zeros(n_neuron)
        self.threshold = threshold
        self._dt = 1/dt

    def __call__(self, x):
        self.d = 1/(1+1e2*torch.abs(x-self.threshold)).to(torch.float)
        s = self._dt*(x>self.threshold)
        return s

    def derivation(self, x):        
        d = 0    
        return d


class spikingWTA(rho):
    def __init__(self, n_neuron, threshold, dt, *args, **kwargs):
        self.threshold = threshold
        self.d = torch.zeros(n_neuron)
        self.defaultd = torch.zeros(n_neuron)
        self._dt = 1/dt

    def __call__(self, x):
        s = 1.*(x>self.threshold)
        if s.sum()>1:
            # Randomly select one index
            s_indices = torch.where(s)[0]
            random_index = s_indices[torch.randint(0, len(s_indices), (1,))]
            s*=0
            s[random_index]=self._dt
    
        self.d = s
        
        return s

    def derivation(self, x):        
        d = 0    
        return d


class probSpiking(rho):
    def __init__(self, n_neuron, scale=0.3, shift=-50., maxprob=0.6, *args, **kwargs):
        self.n_neuron = n_neuron
        self.d = torch.zeros(n_neuron)
        self.defaultd = torch.zeros(n_neuron)
        self.scale = scale
        self.shift = shift
        self.maxprob = maxprob

    def __call__(self, x):
        prob = torch.sigmoid((x-self.shift)*self.scale)
        self.d = self.maxprob*self.scale*prob*(1.-prob)
        s = 1.*(torch.rand(self.n_neuron) < self.maxprob*prob)
        
        return s

    def derivation(self, x):        
        d = 0    
        return d


class probSpikingWTA(rho):
    def __init__(self, n_neuron, scale=0.25, shift=-50., maxprob=0.8, threshold=-60, *args, **kwargs):
        self.n_neuron = n_neuron
        self.d = torch.zeros(n_neuron)
        self.defaultd = torch.zeros(n_neuron)
        self.scale = scale
        self.shift = shift
        self.maxprob = maxprob
        self.threshold = threshold

    def __call__(self, x):
        self.d = torch.zeros(self.n_neuron)
        s = torch.zeros(self.n_neuron)
        if (x<self.threshold).all():
            return s
        max_v, winner = x.max(dim=0)
        prob = torch.sigmoid((max_v-self.shift)*self.scale)
        self.d[winner] = self.maxprob*self.scale*prob*(1.-prob)
        s[winner] = 1.*(torch.rand(1) < self.maxprob*prob)
        
        return s

    def derivation(self, x):        
        d = 0    
        return d


RHO = {
    'linear': linear,
    'relu': relu,
    'sigmoid': sigmoid,
    'softplus': softplus,
    'wta': wta,
    'spiking': spiking,
    'spikingWTA': spikingWTA,
    'probSpiking': probSpiking,
    'probSpikingWTA': probSpikingWTA,
    'tanh': tanh,
}