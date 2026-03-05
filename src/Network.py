import torch
from Neurons.FwdNeuron import *
from Neurons.DeepEligNeuron import *
from Neurons.RecNeuron import *

class FwdNetwork(torch.nn.Module):

    def __init__(self, net=None, layers=None, beta=1., learn_depth=0, dt=0.5, device="cpu"):
        super().__init__()
        self.device = device
        self.beta = beta
        self.dt = dt
        self.learn_depth = int(learn_depth)
        if net is None:
            self.layers = layers
        else: # initial from other network. layer is the class to use.
            self.beta = net.beta
            self.layers = torch.nn.ModuleList()
            for i, l in enumerate(net.layers[:-1]):
                self.layers.append(layers(n_in=l.n_in, n_neurons=l.n_neurons, tau=l.tau, lr_w=l.lr_w, bias=l.bias.detach().clone(),
                      W_in=l.W_in.detach().clone(), activation=l.activation, dt=l.dt, scale=l.scale))
            
            l=net.layers[-1]
            self.layers.append(FwdNeurons(n_in=l.n_in, n_neurons=l.n_neurons, tau=l.tau, lr_w=l.lr_w, bias=l.bias.detach().clone(),
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
        self.layers[-1].E_trg(e_trg=self.beta*error*self.dt)
        for l in reversed(self.layers[self.learn_depth:]):
            l.prop(learn=learn)


    def backwards(self,):
        for l in reversed(self.layers[self.learn_depth:]):
            l.backwards()

    def backwardsRL(self, delta, gamma, labd=1.):
        for l in reversed(self.layers):
            l.backwardsRL(delta, gamma, labd)
    
    def epsilon(self, layer=None):
        if layer is not None:
            return self.layers[layer].epsilon.clone().detach()
        else:
            return [l.epsilon for l in self.layers] 
            
    def learnW(self, update=True):
        for i, l in enumerate(self.layers[self.learn_depth:]):
            l.learnW(update)

    def reset(self, ):
        self.zero_grad(set_to_none=False)
        for i, l in enumerate(self.layers):
            l.reset()

    def align(self, target):
        for i, l in enumerate(self.layers):
            l.u_bar = target.layers[i].u_bar.detach().clone()

    def align0(self,):
        for i, l in enumerate(self.layers):
            l.u_bar *= 0.

        

class DEFwdNetwork(FwdNetwork):
    def __init__(self, net=None, layers=None, beta=1., dt=0.5, device="cpu"):
        super().__init__(net=net, layers=layers, beta=beta, dt=dt, device=device)
        self.initKP()

    def initKP(self, ):
    #initial P and mask for sequential network. Network with recurrrent or skip connection should be redefined.
        #Collect all tau in network
        self.Tau = []
        totaln = 0
        for l in reversed(self.layers):
            l.downstream = totaln
            if l.LP:
                tau_unique, inv, repeat_tau = torch.unique(l.tau, sorted=False, return_inverse=True, return_counts=True)
                l.register_buffer("tau_unique", tau_unique)
                l.register_buffer("dt_tau_uniq", l.dt/l.tau_unique)
                l.register_buffer("decay_uniq", 1 - l.dt_tau_uniq)
                l.register_buffer("repeat_tau", repeat_tau)
                n_tau = tau_unique.shape[0]
                l.n_tau = n_tau
                if l.previous_layer is not None:
                    self.Tau.append(tau_unique)
                    l.P_ = torch.zeros(l.n_neurons, n_tau)
                    l.P_[torch.arange(l.n_neurons), inv] = 1.
                    totaln += n_tau

        self.Tau = torch.hstack(self.Tau)
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




class RTRLNetwork(FwdNetwork):
    def __init__(self, net=None, layers=None, beta=1., dt=0.5, device="cpu"):
        super().__init__(net=net, layers=FwdRFNeurons, beta=beta, dt=dt, device=device)

    def reset(self, batch=1):
        for l in self.layers:
            l.reset()
        for i, l in enumerate(self.layers[:-2]):
            l.P = {}
            l.P_bias = {}
            l.I = torch.eye(l.n_neurons, device=self.device)
            l.P_bias[0] = torch.ones(batch, l.n_neurons, l.n_neurons) * l.I[None, :, :]
            for j, l_ in enumerate(self.layers[i+1:-1]):
                l.P[j+1] = torch.zeros(batch, l.n_neurons, l.n_in, l_.n_neurons)
                l.P_bias[j+1] = torch.zeros(batch, l.n_neurons, l_.n_neurons)
            
    def prop(self, error=0., learn=True):
        self.layers[-1].E_trg(e_trg=self.beta*error*self.dt)
        self.layers[-1].prop()
        self.layers[-2].prop()
        self.last_epsi = self.layers[-2].epsilon
        for i, l in enumerate(self.layers[:-2]): #W_l
            rhod = l.rho.d
            l.P[0] = l.r_bar[:, :, :, None] * l.I[None, :, None, :]   # [B, n_l, n_in, 1]
            for j, l_ in enumerate(self.layers[i+1:-1]): #du_l_/dW
                l.P[j+1] = l_.decay[None, None, None, :] * l.P[j+1] + l_.dt_tau[None, None, 
                    None, :] * ((l.P[j] * rhod[:, None, None, :]) @ l_.W_in.T)
                l.P_bias[j+1] = l_.decay[None, None, :] * l.P_bias[j+1] + l_.dt_tau[None, 
                    None, :] * ((l.P_bias[j] * rhod[:, None, :]) @ l_.W_in.T)
                rhod = l_.rho.d
            l.P_last = l.P[j+1]
            l.P_bias_last = l.P_bias[j+1]
            
            

    def learnW(self, update=True):
        self.layers[-1].learnW(update)
        self.layers[-2].learnW(update)
        for i, l in enumerate(self.layers[:-2]):
            l.dW_in += (l.P_last * self.last_epsi).sum(-1).mean(0)
            l.dbias += (l.P_bias_last * self.last_epsi).sum(-1).mean(0)
            if update:
                l.W_in += l.dW_in * l.lr_w
                l.bias += l.dbias * l.lr_b
                l.dW_in = torch.zeros(l.n_neurons, l.n_in).to(self.device)
                l.dbias = torch.zeros(l.n_neurons).to(self.device)























        