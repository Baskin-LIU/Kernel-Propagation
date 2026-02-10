import torch
from Neurons.FwdNeuron import *
from Neurons.DeepEligNeuron import *
from Neurons.RecNeuron import *

class FwdNetwork(torch.nn.Module):

    def __init__(self, net=None, layers=None, beta=1., dt=0.5, device="cpu"):
        super().__init__()
        self.device=device
        self.beta=beta
        self.dt = dt
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
        self.layers[-1].E_trg(e_trg=self.beta*error*self.dt)
        for l in reversed(self.layers[:]):
            l.prop(learn=learn)

    def backwards(self,):
        for l in reversed(self.layers[:]):
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
        for i, l in enumerate(self.layers):
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
        super().__init__(net, layers, beta, dt, device)
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



def buildKPNet(model_config, general_config):
    device = general_config["device"]
    LP_size = list(model_config['LP_size'])
    Ins_size = list(model_config['Ins_size'])
    dt = general_config['dt']
    tau = []
    tau_min, tau_max, repeat = model_config['Tau0']
    tau0 = np.logspace(np.log10(tau_min), np.log10(tau_max), LP_size[0]//repeat, dtype=np.float32)
    tau.append(np.repeat(tau0[:, None], repeat))
    for i in range(model_config['num_LP_layers']-1):
        tau_uniq = np.array(model_config['Tau%d'%(i+1)])
        tau.append(np.repeat(tau_uniq[:, None],
                                          LP_size[i+1]//tau_uniq.shape[0]))
    layers = torch.nn.ModuleList()
    prev_n=model_config['n_in']
    
    layers.append(
        FwdDENeurons(
            n_in=prev_n,
            n_neurons=LP_size[0],
            tau=tau[0], 
            activation=model_config["activation"], 
            dt=dt, 
            scale=1.0,
            device=device,
        )
    )
    prev_n=LP_size[0]

    if model_config["reducedNonlinear"]:
        for i in range(model_config['num_LP_layers']-2):
            layers.append(
                FwdDENeuronsReduced(
                    n_in=prev_n,
                    n_neurons=LP_size[i+1],
                    tau=tau[i+1], 
                    activation="linear", 
                    dt=dt, 
                    device=device,
                )
            )
            prev_n=LP_size[i+1]
    else:
        for i in range(model_config['num_LP_layers']-2):
            layers.append(
                FwdDENeurons(
                    n_in=prev_n,
                    n_neurons=LP_size[i+1],
                    tau=tau[i+1], 
                    activation=model_config["activation"], 
                    dt=dt, 
                    scale=0.6,
                    device=device,
                )
            )
            prev_n=LP_size[i+1]

    layers.append(
        LastFwdDENeurons(
            n_in=prev_n, 
            n_neurons=LP_size[i+2], 
            tau=tau[i+2], 
            activation=model_config["activation"], 
            dt=dt, 
            scale=1.0,
            device=device,
            )
    )
    prev_n=LP_size[i+2]
    for i in range(model_config['num_Ins_layers']):
        layers.append(
            FwdInsNeurons(
                n_in=prev_n,
                n_neurons=Ins_size[i],
                activation=model_config["activation"], 
                scale=1.0,
                dt=dt,
                device=device,
            )
        )
        prev_n=Ins_size[i]
                
    layers.append(
        FwdInsNeurons(
            n_in=prev_n, 
            n_neurons=model_config['n_out'],
            activation='linear', 
            dt=dt, 
            scale=1.0,
            device=device,
            )
    )
    
    return DEFwdNetwork(layers=layers, device=device)
