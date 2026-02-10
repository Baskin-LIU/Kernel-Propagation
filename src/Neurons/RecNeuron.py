import torch
from .FwdNeuron import *
from .rho import *

class RecNeurons(Neurons):
    
    def __init__(self, n_in, n_neurons, W_in=None, bias=None, tau=20., 
                 inh=0.1, lr_w=1e-2, activation='linear', dt=1., scale =1., device="cpu",):
        super().__init__(n_in, n_neurons, bias, activation, tau, lr_w, dt, device)
        
        self.inh = inh
        self.scale = scale
        self.rho = RHO[activation](n_neurons, scale=self.scale)
        self.weight_init(W_in, bias)

    def step(self, r_in, noise=0., **kwargs):
        self.r_bar = self.decay[None, :, None] * self.r_bar + self.dt_tau[None, :, None] * torch.cat((r_in, self.r), 1)[:, None, :]
        self.u_bar = (self.r_bar * self.W).sum(-1) + self.bias
        
        self.r = self.rho(self.u_bar)

        return self.r, self.u_bar

    def learnW(self, update=True): 
        self.W += self.dW_in * self.lr_w
        self.bias += self.dbias * self.lr_b
        self.dW_in = torch.zeros(self.W_in.shape).to(self.device)
        self.dbias = torch.zeros(self.n_neurons).to(self.device)

    def backwards(self, ):
        self.W.grad -= self.dW
        self.bias.grad -= self.dbias

    def reset(self,):
        super().reset()
        self.r_bar = torch.zeros(1, self.n_neurons, 1).to(self.device)

    def weight_init(self, W, bias, sparse=False):
        # init weight from last layer / input
        if W is not None:
            self.W = torch.nn.Parameter(W)
        else:
            W_inp = torch.empty(self.n_neurons, self.n_in)
            torch.nn.init.kaiming_normal_(W_inp, mode="fan_in", nonlinearity=self.activation)
            W_rec = torch.empty(self.n_neurons, self.n_neurons)
            if sparse:
                p, g = 0.08, 1.2
                N = W_rec.size(0)
                k = p * N
                mask = (torch.rand_like(W_rec) < p).float()
                W_rec.data.normal_(0.0, g/k**0.5)
                W_rec.data *= mask
                eigvals = torch.linalg.eigvals(W_rec)
                rho = eigvals.abs().max()
                W_rec *= (2 / rho)
            else:
                torch.nn.init.orthogonal_(W_rec, gain=2.)
            
            self.W = torch.nn.Parameter(torch.cat((W_inp, W_rec), 1))

        if bias is not None:
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias = torch.empty(self.n_neurons)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / fan_in**0.5
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # diagonal = torch.diag(-torch.rand(n_neurons))
        # Q = torch.randn(n_neurons, n_neurons)
        # W = Q @ diagonal @ torch.inverse(Q) - self.inh
        
        self.W_in = self.W ## Just for reference

        self.W.grad = torch.zeros(self.n_neurons, self.n_in+self.n_neurons)
        self.bias.grad = torch.zeros(self.n_neurons)


class RecDENeurons(RecNeurons):

    def custom_init(self,):
        tau_unique, inv = torch.unique(self.tau, sorted=False, return_inverse=True)
        n_tau = tau_unique.shape[0]
        self.n_tau = n_tau
        P = torch.zeros(self.n_neurons, n_tau)
        P[torch.arange(self.n_neurons), inv] = 1.
        self.register_buffer("P", P.T)
        self.register_buffer("TauDE", tau_unique)
        self.register_buffer("dt_tau_de", self.dt/tau_unique)
        self.register_buffer("decay_de", 1-self.dt_tau_de)

    def prop(self, learn=True):
        self.epsilon = self.rho.d * self.wTe
        if learn:
            self.dW_in += (self.epsilon.unsqueeze(dim=-1) * self.r_bar).mean(0)
            self.dbias += self.epsilon.mean(0)
            K = (self.P[None, :, :] * self.epsilon[:, None, :]) @ self.W[:, self.n_in:]
            self.dW_in += (K.unsqueeze(-1) * self.elig).sum(1).mean(0)
            self.dbias += (K * self.elig_b).sum(1).mean(0)
        # Update input eligibility trace (batch, n_exp, n_neuron, n_in+n_neuron)
        self.elig = self.decay_de[None,:,None,None] * self.elig + self.dt_tau_de[None,:,None,None] * (self.rho.d[:,None,:,None] * self.r_bar[:,None,:,:])
        # Bias eligibility (batch, n_exp, n_neuron)
        self.elig_b = self.decay_de[None,:,None] * self.elig_b + self.dt_tau_de[None, :, None] * self.rho.d[:, None, :]
        
        return 0, 0

    def reset(self,):
        super().reset()
        self.elig = torch.zeros(1,self.n_tau,self.n_neurons,self.n_in+self.n_neurons).to(self.device)
        self.elig_b = torch.zeros(1,self.n_tau, self.n_neurons).to(self.device)


class RecRFNeurons(RecNeurons):

    def prop(self, learn=True):
        self.epsilon = self.rho.d * self.wTe
        
        if learn:
            self.dW_in += (self.epsilon.unsqueeze(dim=-1) * self.r_bar).mean(0)
            self.dbias += self.epsilon.mean(0)
        
        return 0, 0
        

class RecGLENeurons(RecNeurons):        

    def step(self, r_in, noise=0., **kwargs):
        self.r_bar = self.decay[None, :, None] * self.r_bar + self.dt_tau[None, :, None] * torch.cat((r_in, self.r), 1)[:, None, :]
        self.u_bar = (self.r_bar * self.W).sum(-1) + self.bias
        
        self.r = self.rho(self.u_bar)

        return self.r, self.u_bar
    

    def prop(self, learn=True):
        self.epsilon_past = self.epsilon  
        self.epsilon = (self.W_r.T @ self.mismatch + self.wTe) * self.rho.d
        #epsilon breve
        self.mismatch = self.epsilon + self.tau_dt.squeeze(1)*(self.epsilon-self.epsilon_past)

        self.dW_in += (self.mismatch.unsqueeze(dim=-1) * self.r_bar).mean(0)
        self.dbias += self.mismatch.mean(0)
        return 0, 0