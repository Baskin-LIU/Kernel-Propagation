import torch
from FwdNeuron import *
from DeepEligNeuron import *

class FwdNetwork(torch.nn.Module):

    def __init__(self, net=None, layers=None, beta=1., device="cpu"):
        super().__init__()
        self.device=device
        self.beta=beta
        if net is None:
            self.layers = layers
        else: # initial from other network. layer is the class to use.
            self.layers = torch.nn.ModuleList()
            for i, l in enumerate(net.layers[:-1]):
                self.layers.append(layers(n_in=l.n_in, n_neurons=l.n_neurons, tau=l.tau, lr_w=l.lr_w,
                      W_in=l.W_in.detach().clone(), activation=l.activation, dt=l.dt, scale=l.scale))
            
            l=net.layers[-1]
            self.layers.append(FwdNeurons(n_in=l.n_in, n_neurons=l.n_neurons, tau=l.tau, lr_w=l.lr_w,
                      W_in=l.W_in.detach().clone(), activation=l.activation, dt=l.dt))
                
        self.n_layer = len(self.layers)   
        for i, l in enumerate(self.layers):
            if i != 0:
                l.set_previous_layer(self.layers[i-1])
            if i != self.n_layer-1:
                l.set_next_layer(self.layers[i+1])

    
    def step(self, r_in, noise=0):
        r = r_in
        for i, l in enumerate(self.layers):
            r, u = l.step(r_in=r, noise=noise)

        return r, u

    
    def prop(self, error=0., learn=True):
        self.layers[-1].E_trg(e_trg=self.beta*error)
        for l in reversed(self.layers[:]):
            l.prop(learn=learn)

    def backwards(self,):
        for l in reversed(self.layers[:]):
            l.backwards()

    def backwardsRL(self, delta):
        for l in reversed(self.layers[:]):
            l.backwardsRL(delta)
    
    def epsilon(self, layer=None):
        if layer is not None:
            return self.layers[layer].epsilon.clone().detach()
        else:
            return [l.epsilon for l in self.layers] 
            
    def learnW(self, layers=None):
        for i, l in enumerate(self.layers):
            l.learnW()

    def reset(self, ):
        for i, l in enumerate(self.layers):
            l.reset()

    def align(self, target):
        for i, l in enumerate(self.layers):
            l.u_bar = target.layers[i].u_bar.detach().clone()

    def align0(self,):
        for i, l in enumerate(self.layers):
            l.u_bar *= 0.

class DEFwdNetwork(FwdNetwork):
    def __init__(self, net=None, layers=None, beta=1., device="cpu"):
        super().__init__(net, layers, beta, device)
        self.initKP()

    def initKP(self, ):
    #initial P and mask for sequential network. Network with recurrrent or skip connection should be redefined.
        #Collect all tau in network
        self.Tau = []
        totaln = 0
        for l in reversed(self.layers):
            l.downstream = totaln
            if l.LP and l.previous_layer is not None:
                tau_unique, inv = torch.unique(l.tau, sorted=False, return_inverse=True)
                self.Tau.append(tau_unique)
                n_tau = tau_unique.shape[0]
                l.n_tau = n_tau
                l.P_ = torch.zeros(l.n_neurons, n_tau)
                l.P_[torch.arange(l.n_neurons), inv] = 1.
                totaln += n_tau

                l.register_buffer("tau_unique", tau_unique)
                l.register_buffer("dt_tau_uniq", l.dt/l.tau_unique)
                l.register_buffer("decay_uniq", 1 - l.dt_tau_uniq)

        self.Tau = torch.hstack(self.Tau)
        print(self.Tau)
        assert self.Tau.numel() == torch.unique(self.Tau).numel() #forbid same tau in different layers
        #unlock by the path?
        self.totalN = totaln

        #Initial P
        n_=0
        for l in self.layers:
            l.totalN = self.totalN
            if l.LP and l.previous_layer is not None:
                P = self.Tau[None, :] - l.tau[:, None]
                P = torch.true_divide(self.Tau[None, :], P)
                #for current layer tau
                P[:, l.downstream:l.downstream+l.n_tau] = l.P_
                del l.P_
                l.register_buffer("P", P.T)
            else:
                l.register_buffer("P", torch.tensor(1.))
            # init variables for tracing DeepE
            if l.downstream:
                l.register_buffer("TauDE", self.Tau[:l.downstream])
                l.register_buffer("dt_tau_de", l.dt/l.TauDE)
                l.register_buffer("decay_de", 1-l.dt_tau_de)
            else:
                break 
