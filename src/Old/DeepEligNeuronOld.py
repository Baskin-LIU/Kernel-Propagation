from rho import *
from RateNeuron import *
import numpy as np
import torch

class OutNeurons(Neurons):
    # instantaneous neuron to save computation
    def __init__(self, n_in, n_neurons, W_in=None, tau=20., lr_w = 1e-2, beta=0.1,
                 activation='linear', dt=1., device="cpu", bias=False):
        super().__init__(n_in, n_neurons, bias, activation, tau, lr_w, beta, dt, device)

        self.rho = RHO[activation](n_neurons)
        # init weight from last layer / input
        self.W_in = torch.nn.Parameter(1.5*torch.randn(n_neurons, n_in), requires_grad=False
                                      ) if W_in is None else torch.nn.Parameter(W_in, requires_grad=False)

        self.LP = False
        self.decay = (1-self.dt_tau.unsqueeze(1))
        

    def step(self, r_in, noise=0.):
        self.r_in=r_in
        #update u and output
        self.u_d = (self.W_in * self.r_in).sum(dim=1) + self.b
        self.u_bar = self.u_d
        self.r = self.u_bar

        return self.r, self.u_bar

    def prop(self, e_trg=0):
        
        self.mismatch = self.epsilon_inst = self.epsilon_lpf = self.epsilon_lpf_bar = self.beta * e_trg
        self.previous_layer.wTe = self.W_in.T @ self.mismatch
        self.previous_layer.epsilon_lpf = self.W_in.T @ self.mismatch

        return 0, 0

    def learnW(self,):    
        dW_in = self.mismatch.unsqueeze(dim=1) * self.r_in
        self.W_in += dW_in * self.dt * self.lr_w

        if self.bias:
            self.b += self.mismatch * self.dt * self.lr_w

class FwdDEL2Neurons(Neurons):
    
    def __init__(self, n_in, n_neurons, n_a=None, u_rest=0., W_in=None, tau_d=20., lr_w = 1e-2, lr_p = 1e-1, beta=0.1,
                 activation='linear', afunc='inp', dt=1., device="cpu",):
        super().__init__(n_in, n_neurons, n_a, u_rest, tau_d, lr_w, lr_p, beta, dt, activation, afunc, device)

        self.rho = RHO[activation](n_neurons, scale=0.5)
        # init weight from last layer / input
        self.W_in = torch.nn.Parameter(torch.randn(n_neurons, n_in), requires_grad=False
                                      ) if W_in is None else torch.nn.Parameter(W_in, requires_grad=False)

        self.r_bar = torch.zeros(self.n_neurons, self.n_in) #r_in_bar
        self.elig = torch.zeros(self.n_neurons, self.n_in)
        self.decay = 1-self.dt_tau
        self.dt_tau_de = torch.ones(self.n_neurons)
        self.epsilon_inst = torch.zeros(self.n_neurons)
        self.epsilon_lpf = torch.zeros(self.n_neurons)
        self.epsilon_lpf_bar = torch.zeros(self.n_neurons)
        self.epsilon_lpf_past = torch.zeros(self.n_neurons)
        self.epsilon_lpf_dot = torch.zeros(self.n_neurons)

    def step(self, r_in, noise=0.):
        self.r_in=r_in
        self.r_bar = self.decay.unsqueeze(1)*self.r_bar + torch.outer(self.dt_tau, self.r_in)

        #update u and output
        self.u_d = (self.W_in * self.r_in).sum(dim=1)

        if self.previous_layer != None:
            self.previous_layer.wTe = self.W_in.T @ self.mismatch

        self.u_bar = self.u_bar + self.dt_tau*(self.u_rest - self.u_bar + self.u_d) + noise
        self.r = self.rho(self.u_bar)

        return self.r, self.u_bar

    def learnP(self, e_trg=0.):  
        self.mismatch = self.epsilon_inst = self.wTe
        self.epsilon_lpf_bar = self.decay*self.epsilon_lpf_bar + self.dt_tau * self.epsilon_lpf
        
        if self.previous_layer != None:
            self.previous_layer.wTe = self.W_in.T @ (self.rho.d*self.dt_tau*self.epsilon_inst)
            self.previous_layer.epsilon_lpf = self.W_in.T @ (self.rho.d*self.epsilon_lpf_bar)

        self.epsilon_lpf_dot = self.epsilon_lpf - self.epsilon_lpf_past
        if ((self.epsilon_inst - self.epsilon_lpf) != 0).all():
            #self.dt_tau_de = (self.epsilon_lpf_dot/(self.epsilon_inst - self.epsilon_lpf))
            self.dt_tau_de = torch.clamp((self.epsilon_inst - self.epsilon_lpf_dot)/self.epsilon_lpf, 0, 1)
        
        self.elig = (1-self.dt_tau_de.unsqueeze(1))*self.elig + self.rho.d.unsqueeze(1) * self.r_bar # *self.A
        self.epsilon_lpf_past = self.epsilon_lpf
        return 0, 0

    def learnW(self,):
        dW_in = self.mismatch.unsqueeze(dim=1) * self.elig
        self.W_in += dW_in * self.dt * self.lr_w
        if self.bias: # how is bias being lpf ？？？？？？？？
            self.b += self.mismatch * self.dt * self.lr_w  * 0.5


    
class LastNeurons(Neurons):
    # L-1 neurons before inst output to save computation
    def __init__(self, n_in, n_neurons, n_a=None, u_rest=0., W_in=None, tau_d=20., lr_w = 1e-2, lr_p = 1e-1, beta=0.1,
                 activation='linear', afunc='inp', dt=1., device="cpu", bias=False):
        super().__init__(n_in, n_neurons, n_a, u_rest, tau_d, lr_w, lr_p, beta, dt, activation, afunc, device, bias)

        self.rho = RHO[activation](n_neurons, scale=0.8)
        # init weight from last layer / input
        self.W_in = torch.nn.Parameter(1.5*torch.randn(n_neurons, n_in), requires_grad=False
                                      ) if W_in is None else torch.nn.Parameter(W_in, requires_grad=False)

        self.r_bar = torch.zeros(self.n_neurons, self.n_in) #r_in_bar
        self.elig = torch.zeros(self.n_neurons, self.n_in)
        self.decay = 1-self.dt_tau
        

    def step(self, r_in, noise=0.):
        self.r_in=r_in
        self.r_bar = self.decay.unsqueeze(1)*self.r_bar + torch.outer(self.dt_tau, self.r_in)
        #n_neur*n_exp*n_in

        #update u and output
        self.u_d = (self.W_in * self.r_in).sum(dim=1) + self.b

        self.u_bar = self.u_bar + self.dt_tau*(self.u_rest - self.u_bar + self.u_d) + noise
        self.r = self.rho(self.u_bar)

        return self.r, self.u_bar

    def learnP(self, e_trg=0):
        self.mismatch = self.rho.d*self.wTe
        Cl = torch.tile(self.wTe.unsqueeze(1), (1, self.totalN))
        self.Cl = self.W_in.T @ (self.rho.d.unsqueeze(1)*Cl*self.denom)
        self.elig = self.r_bar

        return 0, 0

    def learnW(self,):
        dW_in = self.mismatch.unsqueeze(dim=1) * self.elig
        self.W_in += dW_in * self.dt * self.lr_w
        if self.bias:
            self.b += self.mismatch * self.dt * self.lr_w



class FwdDENeurons(Neurons):
    
    def __init__(self, n_in, n_neurons, n_a=None, u_rest=0., W_in=None, tau_d=20., lr_w = 1e-2, lr_p = 1e-1, beta=0.1,
                 activation='linear', dt=1., device="cpu", bias=False):
        super().__init__(n_in, n_neurons, n_a, u_rest, tau_d, lr_w, lr_p, beta, dt, activation, None, device, bias)

        self.rho = RHO[activation](n_neurons, scale=0.8)
        # init weight from last layer / input
        self.W_in = torch.nn.Parameter(1.5*torch.randn(n_neurons, n_in), requires_grad=False
                                      ) if W_in is None else torch.nn.Parameter(W_in, requires_grad=False)

        self.r_bar = torch.zeros(self.n_neurons, self.n_in) #r_in_bar
        self.elig = torch.zeros(self.n_neurons, self.n_in)
        self.decay = 1-self.dt_tau
        

    def step(self, r_in, noise=0.):
        self.r_in=r_in
        self.r_bar = self.decay.unsqueeze(1)*self.r_bar + torch.outer(self.dt_tau, self.r_in)

        #update u and output
        self.u_d = (self.W_in * self.r_in).sum(dim=1) + self.b

        self.u_bar = self.u_bar + self.dt_tau*(self.u_rest - self.u_bar + self.u_d) + noise
        self.r = self.rho(self.u_bar)

        self.elig = (1-self.dt_tau_de)*self.elig + self.dt_tau_de*(self.rho.d.unsqueeze(1)*self.r_bar).unsqueeze(1)
        #n_neur*n_exp
        self.elig_b = (1-self.dt_tau_de.squeeze(-1))*self.elig_b + self.dt_tau_de.squeeze(-1) * self.rho.d.unsqueeze(1)

        return self.r, self.u_bar

    def learnP(self, e_trg=0.):
        self.C = self.next_layer.Cl[:, self.mask:] #mask not include itself
        self.Cl = self.W_in.T @ (self.rho.d.unsqueeze(1)*self.next_layer.Cl*self.denom)
        
        return 0, 0

    def learnW(self,):
        self.dW_in = (self.C.unsqueeze(-1)*self.elig).sum(axis=1)
        self.W_in += self.dW_in * self.dt * self.lr_w
        if self.bias: # how is bias being lpf ？？？？？？？？
            self.b += (self.C*self.elig_b).sum(axis=1) * self.dt * self.lr_w


class RecDENeurons(Neurons):
    
    def __init__(self, n_in, n_neurons, n_a=None, u_rest=0., W_in=None, tau_d=20., lr_w = 1e-2, lr_p = 1e-1, beta=0.1,
                 activation='linear', dt=1., device="cpu", bias=False):
        super().__init__(n_in, n_neurons, n_a, u_rest, tau_d, lr_w, lr_p, beta, dt, activation, None, device, bias)

        self.rho = RHO[activation](n_neurons, scale=0.8)
        # init weight from last layer / input
        self.W_in = torch.nn.Parameter(1.5*torch.randn(n_neurons, n_in), requires_grad=False
                                      ) if W_in is None else torch.nn.Parameter(W_in, requires_grad=False)

        self.r_bar = torch.zeros(self.n_neurons, self.n_in) #r_in_bar
        self.elig = torch.zeros(self.n_neurons, self.n_in)
        self.decay = 1-self.dt_tau
        

    def step(self, r_in, noise=0.):
        self.r_in=r_in
        self.r_bar = self.decay.unsqueeze(1)*self.r_bar + torch.outer(self.dt_tau, self.r_in)

        #update u and output
        self.u_d = (self.W_in * self.r_in).sum(dim=1) + self.b

        self.u_bar = self.u_bar + self.dt_tau*(self.u_rest - self.u_bar + self.u_d) + noise
        self.r = self.rho(self.u_bar)

        self.elig = (1-self.dt_tau_de)*self.elig + self.dt_tau_de*(self.rho.d.unsqueeze(1)*self.r_bar).unsqueeze(1)
        #n_neur*n_exp
        self.elig_b = (1-self.dt_tau_de.squeeze(-1))*self.elig_b + self.dt_tau_de.squeeze(-1) * self.rho.d.unsqueeze(1)

        return self.r, self.u_bar

    def learnP(self, e_trg=0.):
        self.mismatch = self.rho.d*self.wTe
        Cl = torch.tile(self.wTe.unsqueeze(1), (1, self.totalN))

        self.Cl = self.W_in.T @ (self.rho.d.unsqueeze(1)*Cl*self.denom)
        self.C = self.Cl

        self.Cl = self.W_in.T @ (self.rho.d.unsqueeze(1)*Cl*self.denom)
        self.C += self.Cl
        
        self.elig = self.r_bar
        
        self.C = self.next_layer.Cl[:, self.mask:] #mask not include itself
        self.Cl = self.W_in.T @ (self.rho.d.unsqueeze(1)*self.next_layer.Cl*self.denom)
        
        return 0, 0

    def learnW(self,):
        self.dW_in = (self.C.unsqueeze(-1)*self.elig).sum(axis=1)
        self.W_in += self.dW_in * self.dt * self.lr_w
        if self.bias: # how is bias being lpf ？？？？？？？？
            self.b += (self.C*self.elig_b).sum(axis=1) * self.dt * self.lr_w