import torch
from Neurons.FwdNeuron import *
from Neurons.DeepEligNeuron import *
from Neurons.RecNeuron import *

SKIP_WEIGHT = 0.2

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
            
            l = net.layers[-1]
            self.layers.append(FwdNeurons(n_in=l.n_in, n_neurons=l.n_neurons, tau=l.tau, lr_w=l.lr_w, bias=l.bias.detach().clone(),
                      W_in=l.W_in.detach().clone(), activation=l.activation, dt=l.dt, scale=l.scale))
                
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
            if i < self.learn_depth:
                r = r.detach()
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
            l.register_buffer("downstream_mask", torch.arange(totaln, dtype=torch.int32))
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


class DESkipNetwork(DEFwdNetwork):
    def __init__(self, net=None, layers=None, beta=1., dt=0.5, device="cpu"):
        super().__init__(net=net, layers=layers, beta=beta, dt=dt, device=device)
        l1_ntau = self.layers[1].tau_unique.shape[0]
        self.mask = torch.cat((self.layers[3].downstream_mask, 
                                                       torch.arange(l1_ntau)+self.layers[1].downstream))
        self.skip_weight = SKIP_WEIGHT #??

    def prop(self, error=0., learn=True):      
        self.layers[6].E_trg(e_trg=self.beta*error*self.dt)
        self.layers[6].prop(learn=learn)
        self.layers[5].prop(learn=learn)
        self.layers[4].prop(learn=learn)
        self.layers[3].prop(learn=learn)
        self.layers[2].prop(learn=learn)
        if learn:
            K4_1 = torch.zeros_like(self.layers[3].K)
            K4_1[:, self.mask] = self.layers[3].K[:, self.mask]
            self.layers[1].K += K4_1 * self.skip_weight
        self.layers[1].prop(learn=learn)
        self.layers[0].prop(learn=False)

    def step(self, r_in, noise=0):
        r0,_ = self.layers[0].step(r_in=r_in)
        r1,_ = self.layers[1].step(r_in=r0)
        r2,_ = self.layers[2].step(r_in=r1)
        r3,_ = self.layers[3].step(r_in=r2)
        r4,_ = self.layers[4].step(r_in=r1*self.skip_weight+r3)
        r5,_ = self.layers[5].step(r_in=r4) #ins
        r,u  = self.layers[6].step(r_in=r5)

        return r, u


class SkipNetwork(FwdNetwork):
    def __init__(self, net=None, layers=None, beta=1., dt=0.5, device="cpu"):
        super().__init__(net=net, layers=layers, beta=beta, dt=dt, device=device)
        self.layers[2].rho.scale = 1.
        self.skip_weight = SKIP_WEIGHT

    def prop(self, error=0., learn=True):      
        self.layers[6].E_trg(e_trg=self.beta*error*self.dt)
        self.layers[5].prop(learn=learn)
        self.layers[4].prop(learn=learn)
        self.layers[3].prop(learn=learn)
        self.layers[2].prop(learn=learn)
        if learn:
            self.layers[1].epsilon += self.layers[3].epsilon * self.skip_weight
        self.layers[1].prop(learn=learn)
        self.layers[0].prop(learn=False)

    def step(self, r_in, noise=0):
        r0,_ = self.layers[0].step(r_in=r_in)
        r1,_ = self.layers[1].step(r_in=r0)
        r2,_ = self.layers[2].step(r_in=r1)
        r3,_ = self.layers[3].step(r_in=r2)
        r4,_ = self.layers[4].step(r_in=r1*self.skip_weight+r3)
        r5,_ = self.layers[5].step(r_in=r4) #ins
        r,u  = self.layers[6].step(r_in=r5)

        return r, u


class DESkipNetworkDetach(DEFwdNetwork):
    def __init__(self, net=None, layers=None, beta=1., dt=0.5, device="cpu"):
        super().__init__(net=net, layers=layers, beta=beta, dt=dt, device=device)
        l1_ntau = self.layers[1].tau_unique.shape[0]
        self.layers[0].downstream = self.layers[3].downstream + l1_ntau
        self.layers[0].downstream_mask = torch.cat((self.layers[3].downstream_mask, 
                                                       torch.arange(l1_ntau)+self.layers[1].downstream))
        self.layers[0].TauDE = torch.cat((self.layers[3].TauDE, self.layers[1].tau_unique))
        self.layers[0].dt_tau_de = self.dt/self.layers[0].TauDE
        self.layers[0].decay_de = 1-self.layers[0].dt_tau_de
        
        self.layers[1].downstream = self.layers[3].downstream
        self.layers[1].downstream_mask = self.layers[3].downstream_mask
        self.layers[1].TauDE = self.layers[3].TauDE
        self.layers[1].dt_tau_de = self.layers[3].dt_tau_de
        self.layers[1].decay_de = self.layers[3].decay_de

        self.layers[2].rho.scale = 1.
        self.skip_weight = SKIP_WEIGHT

    def prop(self, error=0., learn=True):      
        self.layers[6].E_trg(e_trg=self.beta*error*self.dt)
        self.layers[6].prop(learn=learn)
        self.layers[5].prop(learn=learn)
        self.layers[4].prop(learn=learn)
        self.layers[3].prop(learn=learn)
        self.layers[2].prop(learn=False)
        if learn:
            self.layers[1].K = self.layers[3].K * self.skip_weight
        if self.learn_depth==0:
            self.layers[1].prop(learn=learn)
            self.layers[0].prop(learn=False)
        else:
            self.layers[1].prop(learn=False)

    def step(self, r_in, noise=0):
        r0,_ = self.layers[0].step(r_in=r_in)
        r1,_ = self.layers[1].step(r_in=r0)
        r2,_ = self.layers[2].step(r_in=r1.detach())
        r3,_ = self.layers[3].step(r_in=r2)
        r4,_ = self.layers[4].step(r_in=r1*self.skip_weight+r3)
        r5,_ = self.layers[5].step(r_in=r4) #ins
        r,u  = self.layers[6].step(r_in=r5)

        return r, u


class SkipNetworkDetach(FwdNetwork):
    def __init__(self, net=None, layers=None, beta=1., dt=0.5, device="cpu"):
        super().__init__(net=net, layers=layers, beta=beta, dt=dt, device=device)
        self.layers[2].rho.scale = 1.
        self.skip_weight = SKIP_WEIGHT

    def prop(self, error=0., learn=True):      
        self.layers[6].E_trg(e_trg=self.beta*error*self.dt)
        self.layers[5].prop(learn=learn)
        self.layers[4].prop(learn=learn)
        if self.learn_depth==4:
            return
        self.layers[3].prop(learn=learn)
        self.layers[2].prop(learn=False)
        if learn:
            self.layers[1].epsilon = self.layers[3].epsilon * self.skip_weight
        self.layers[1].prop(learn=learn)
        self.layers[0].prop(learn=False)

    def step(self, r_in, noise=0):
        r0,_ = self.layers[0].step(r_in=r_in)
        r1,_ = self.layers[1].step(r_in=r0)
        r1_detached = r1.clone().detach().requires_grad_(False)
        r2,_ = self.layers[2].step(r_in=r1_detached)
        r3,_ = self.layers[3].step(r_in=r2)
        r4,_ = self.layers[4].step(r_in=r1*self.skip_weight+r3)
        r5,_ = self.layers[5].step(r_in=r4) #ins
        r,u  = self.layers[6].step(r_in=r5)

        return r, u


class DEAllSkipNetwork(DEFwdNetwork):
    def __init__(self, net=None, layers=None, beta=1., dt=0.5, device="cpu"):
        super().__init__(net=net, layers=layers, beta=beta, dt=dt, device=device)
        self.layers[4].P = self.layers[4].P[:self.layers[4].n_tau]
        for i in [0, 1, 2]:
            self.layers[i].downstream = self.layers[3].downstream
            self.layers[i].downstream_mask = self.layers[3].downstream_mask
            self.layers[i].TauDE = self.layers[3].TauDE
            self.layers[i].dt_tau_de = self.layers[3].dt_tau_de
            self.layers[i].decay_de = self.layers[3].decay_de
            self.layers[i].rho.scale = 1.


    def prop(self, error=0., learn=True):      
        self.layers[6].E_trg(e_trg=self.beta*error*self.dt)
        self.layers[6].prop(learn=learn)
        self.layers[5].prop(learn=learn)
        self.layers[4].prop(learn=learn)
        self.layers[3].prop(learn=False)
        self.layers[2].prop(learn=False)
        self.layers[1].prop(learn=False)
        self.layers[0].prop(learn=False)
        if learn:
            self.layers[0].K = self.layers[3].K * 0.2
            self.layers[1].K = self.layers[3].K * 0.3
            self.layers[2].K = self.layers[3].K * 0.4
            self.layers[3].K = self.layers[3].K * 0.5

    def step(self, r_in, noise=0):
        r0,_ = self.layers[0].step(r_in=r_in)
        r1,_ = self.layers[1].step(r_in=r0.detach())
        r2,_ = self.layers[2].step(r_in=r1.detach())
        r3,_ = self.layers[3].step(r_in=r2.detach())
        r4,_ = self.layers[4].step(r_in=r3*0.5+r2*0.4+r1*0.3+r0*0.2)
        r5,_ = self.layers[5].step(r_in=r4) #ins
        r,u  = self.layers[6].step(r_in=r5)

        return r, u


class AllSkipNetwork(FwdNetwork):
    def __init__(self, net=None, layers=None, beta=1., dt=0.5, device="cpu"):
        super().__init__(net=net, layers=layers, beta=beta, dt=dt, device=device)
        for l in self.layers:
            l.rho.scale = 1.

    def step(self, r_in, noise=0):
        
        r_prev = r_in
        weight = 0.2
        r_in_L = 0
        
        for l in self.layers[:-3]:
            r,_ = l.step(r_in=r_prev)
            r_in_L += weight * r
            r_prev = r.detach()
            weight += 0.1
        r,_ = self.layers[-3].step(r_in_L)
        r,_ = self.layers[-2].step(r)
        r,u = self.layers[-1].step(r)

        return r, u
            

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
        Grads = []
        for i, l in enumerate(self.layers[:-2]):
            l.dW_in += (l.P_last * self.last_epsi).sum(-1).mean(0)
            l.dbias += (l.P_bias_last * self.last_epsi).sum(-1).mean(0)
            if update:
                l.W_in += l.dW_in * l.lr_w
                l.bias += l.dbias * l.lr_b
                grad = [l.dW_in.clone(), l.dbias.clone()]
                Grads.append(grad)
                l.dW_in = torch.zeros(l.n_neurons, l.n_in).to(self.device)
                l.dbias = torch.zeros(l.n_neurons).to(self.device)
        grad = self.layers[-2].learnW(update)
        Grads.append(grad)
        grad = self.layers[-1].learnW(update)
        Grads.append(grad)
        return Grads


class UORONetwork(FwdNetwork):
    def __init__(self, net=None, layers=None, beta=1., dt=0.5, n_LP=3, device="cpu"):
        super().__init__(net=net, layers=FwdRFNeurons, beta=beta, dt=dt, device=device)
        self.n_LP = n_LP
        self.n_Ins = len(self.layers) - self.n_LP
        self.total_neurons = sum(l.n_neurons for l in self.layers[:self.n_LP])
        self.total_weights = self.total_weights = sum(l.W_in.numel() + l.n_neurons for l in self.layers[:self.n_LP])
        
    def reset(self, batch=1):
        for l in self.layers:
            l.reset()
        self.utilde = torch.zeros(1, self.total_neurons, device=self.device)
        self.thetatilde = torch.zeros(1, self.total_weights, device=self.device)
            
    def prop(self, error=0., learn=True):
        self.batch = self.layers[0].r.shape[0]
        self.layers[-1].E_trg(e_trg=self.beta*error*self.dt)
        for i in range(self.n_Ins+1):
            self.layers[-i-1].prop()
            
        nu = torch.randint(0, 2, self.utilde.shape, device=self.device).float() * 2 - 1
        utilde_new = torch.zeros_like(self.utilde)
        offset, next_term = self.total_neurons, 0
    
        # -------- utilde propagation ----------
        for l in reversed(self.layers[:self.n_LP]):
            n = l.n_neurons
            idx = slice(offset-n, offset)
            # decay
            utilde_new[:, idx] = l.decay * self.utilde[:, idx]
            utilde_new[:, idx] += l.rho.d * next_term
            if l.previous_layer is not None:
                next_term = l.dt_tau * self.utilde[:, idx] @ l.W_in.T
            offset -= n
    
        # -------- compute P^T nu --------------
        ptTnu = torch.zeros_like(self.thetatilde)
        offset_n, offset_w = 0, 0
    
        for l, layer in enumerate(self.layers[:self.n_LP]):
            n = layer.n_neurons
            m = layer.W_in.shape[1]
    
            idx_n = slice(offset_n, offset_n+n)
    
            nu_l = nu[:, idx_n]
            grad_w = (
                nu_l[:, :, None] * layer.r_in[:, None, :] * layer.dt_tau[None, :, None] 
            ).reshape(self.batch, -1)
            grad_b = 1. * nu_l # bias gradient
            grad = torch.cat([grad_w, grad_b], dim=1)
    
            size = grad.shape[1]
            ptTnu[:, offset_w:offset_w+size] = grad
    
            offset_w += size
            offset_n += n
    
        # -------- rank-1 update ---------------
        eps = 1e-7
        norm_s = utilde_new.norm(dim=1, keepdim=True) + eps
        norm_nu = nu.norm(dim=1, keepdim=True) + eps
        rho0 = torch.sqrt(self.thetatilde.norm(dim=1, keepdim=True) / norm_s + eps)
        rho1 = torch.sqrt(ptTnu.norm(dim=1, keepdim=True) / norm_nu + eps )
        
        self.utilde = rho0 * utilde_new + rho1 * nu
        self.thetatilde = self.thetatilde / rho0 + ptTnu / rho1

        #print(self.utilde, self.thetatilde, rho0, rho1)

    def learnW(self, update=True):
        last_LP = self.layers[self.n_LP-1]
        dLdu = last_LP.epsilon
        grad = (dLdu * self.utilde[:, -last_LP.n_neurons:]).sum(-1, keepdim=True) * self.thetatilde
        grad = torch.clamp(grad.mean(0), -0.05, 0.05)
        offset_w = 0
        for i, l in enumerate(self.layers):
            if i < self.n_LP:
                n_w, n_b = l.W_in.numel(), l.n_neurons
                l.dW_in += grad[offset_w : offset_w + n_w].reshape(l.dW_in.shape)
                l.dbias += grad[offset_w + n_w : offset_w + n_w + n_b]
                offset_w += n_w + n_b
                if update:
                    #print(l.dW_in, l.dbias)
                    l.W_in += l.dW_in * l.lr_w * 2
                    l.bias += l.dbias * l.lr_b * 2
                    l.dW_in = torch.zeros(l.n_neurons, l.n_in).to(self.device)
                    l.dbias = torch.zeros(l.n_neurons).to(self.device)
            else:
                l.learnW(update)









        