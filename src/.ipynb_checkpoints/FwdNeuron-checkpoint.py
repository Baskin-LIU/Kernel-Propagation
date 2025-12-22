from rho import *
import numpy as np
import torch

class Neurons(torch.nn.Module):

    def __init__(self, n_in, n_neurons, bias=False, activation='linear', tau=20., lr_w=1e-2, 
                 beta=1., dt=1., device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.beta = beta
        self.n_in = n_in
        self.n_neurons = n_neurons #n_out
        self.bias = bias
        self.b = torch.zeros(n_neurons)
        
        self._init_tau(tau)
        self.lr_w = lr_w
        self.lr_b = lr_w*1e-1
        self.dt = dt

        self.dt_tau = self.dt/self.tau
        self.tau_dt = (self.tau/self.dt).unsqueeze(1)
        self.tau_dt[self.tau_dt==1] = 0.
        self.decay = 1-self.dt_tau

        self.next_layer = None
        self.previous_layer = None

        self.u_d = torch.zeros(n_neurons, )
        self.u_bar = torch.zeros(n_neurons, )
        self.r = torch.zeros(n_neurons, ) #output rate
        self.wTe = torch.zeros(n_neurons)
        self.r_in = torch.zeros(n_in)
        self.activation = activation
        self.mismatch = torch.zeros(n_neurons)
        self.epsilon = torch.zeros(n_neurons)

        self.rho = rho(n_neurons)
        
        #self.dW_in_bar = torch.zeros(n_neurons, n_in)

        self.custom_init()

    def custom_init(self, ):
        return
    
    def step(self,):
        return

    def set_next_layer(self, nxt):
        if self.next_layer is None:
            self.next_layer = nxt

    def set_previous_layer(self, previous):
        if self.previous_layer is None:
            self.previous_layer = previous
        
    def reset(self, ):
        self.u_bar *= 0
        self.rho.reset()

    def _init_tau(self, tau):
        if type(tau) is tuple: #random initialize within the given range
            self.tau_min, self.tau_max = tau
            self.tau= torch.rand(self.n_neurons)*(self.tau_max - self.tau_min) + self.tau_min
        elif type(tau) is float or type(tau) is int: #scalar tau then apply same for all
            self.tau= torch.tensor(tau).repeat(self.n_neurons)
        elif type(tau) is list: #assigned tau
            self.tau= torch.tensor(tau)
        elif isinstance(tau, (np.ndarray, np.generic)):
            self.tau= torch.tensor(tau, dtype=torch.float32)
        elif isinstance(tau, torch.Tensor): #assigned tau
            self.tau= tau.clone()
        else:
            raise NotImplementedError

        assert len(self.tau) == self.n_neurons




class FwdNeurons(Neurons):
    
    def __init__(self, n_in, n_neurons, bias=False, activation='linear', tau=20., lr_w=1e-2, 
                 W_in=None, scale=0.8, beta=0.1, dt=1., device="cpu"):
        super().__init__(n_in, n_neurons, bias, activation, tau, lr_w, beta, dt, device)
        self.scale = scale
        self.rho = RHO[activation](n_neurons, scale=self.scale)
        # init weight from last layer / input
        self.W_in = torch.nn.Parameter(torch.randn(n_neurons, n_in), requires_grad=False
                                      ) if W_in is None else torch.nn.Parameter(W_in, requires_grad=False)
        self.LP = self.tau.max()>self.dt
        

    def step(self, r_in, noise=0., **kwargs):
        self.r_in = r_in
        #update u and output
        self.u_d = (self.W_in * self.r_in).sum(dim=1) + self.b + self.mismatch# + noise

        self.u_bar = self.decay * self.u_bar + self.dt_tau * self.u_d
        self.r = self.rho(self.u_bar)

        return self.r, self.u_bar
    

    def prop(self,):
        self.epsilon = self.wTe * self.rho.d
        if self.previous_layer != None:
            self.previous_layer.wTe = self.W_in.T @ self.epsilon
        return 0, 0
        
        
    def learnW(self,):    
        dW_in = self.epsilon.unsqueeze(dim=1) * self.r_in
        self.W_in += dW_in * self.dt * self.lr_w
        self.b += self.epsilon * self.dt * self.lr_b

    
    def E_trg(self, e_trg=0,):
        self.epsilon = self.beta * self.rho.d * e_trg
        self.previous_layer.wTe = self.W_in.T @ self.epsilon

        

class FwdInsNeurons(FwdNeurons):
    
    def __init__(self, n_in, n_neurons, bias=False, activation='linear', tau=0., lr_w=1e-2, 
                 W_in=None, scale=0.8, beta=0.1, dt=1., device="cpu"):
        super().__init__(n_in, n_neurons, bias, activation, tau, lr_w, W_in, scale, beta, dt, device) 
        self.LP = False
        

    def step(self, r_in, noise=0., **kwargs):
        self.r_in=r_in
        #update u and output
        self.u_d = (self.W_in * self.r_in).sum(dim=1) + self.mismatch + self.b
        self.u_bar = self.u_d
        self.r = self.rho(self.u_bar)

        return self.r, self.u_d


class FwdGLENeurons(FwdNeurons):

    def custom_init(self,):
        self.mismatch = torch.zeros(self.n_neurons)

    def prop(self,):
        self.epsilon_past = self.epsilon    
        self.epsilon = self.wTe * self.rho.d
        self.mismatch = self.epsilon + self.tau_dt.squeeze(1)*(self.epsilon-self.epsilon_past)
        if self.previous_layer != None:
            self.previous_layer.wTe = self.W_in.T @ self.mismatch
        return 0, 0

        
    def learnW(self,):
        dW_in = self.mismatch.unsqueeze(dim=1) * self.r_in
        self.W_in += dW_in * self.dt * self.lr_w
        self.b += self.mismatch * self.dt * self.lr_b

        

class FwdEPNeurons(FwdNeurons):
    
    def custom_init(self, ):
        self.r_bar = torch.zeros(self.n_neurons, self.n_in) #r_in_bar
        

    def prop(self,):
        super().prop()
        self.r_bar = self.decay[:, None] * self.r_bar + torch.outer(self.dt_tau, self.r_in)
        return 0, 0
        

    def learnW(self,):
        dW_in = self.epsilon.unsqueeze(dim=1) * self.r_bar
        #print(dW_in)
        self.W_in += dW_in * self.dt * self.lr_w
        self.b += self.epsilon * self.dt * self.lr_b


        
class FwdMulNeurons(FwdNeurons):

    #Build Mul-input layer. 
    
    def custom_init(self, ):
        self.n_in_ = self.n_in
        self.n_in = sum(self.n_in)

    
    def prop(self, ):
        self.epsilon = self.wTe * self.rho.d
        wTe = self.W_in.T @ self.epsilon
        n_ = 0
        for i, n in enumerate(self.n_in_):
            self.previous_layer[i].wTe = wTe[n_:n_+n]
            n_ += n
        return err, dP



class FwdDEL2Neurons(Neurons):
    
    def __init__(self, n_in, n_neurons, W_in=None, tau=20., lr_w = 1e-2, beta=0.1,
                 activation='linear', dt=1., device="cpu", bias=False):
        super().__init__(n_in, n_neurons, bias, activation, tau, lr_w, beta, dt, device)

        self.rho = RHO[activation](n_neurons, scale=0.8)
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

        self.u_bar = self.u_bar + self.dt_tau*(- self.u_bar + self.u_d) + noise
        self.r = self.rho(self.u_bar)

        return self.r, self.u_bar

    def prop(self, e_trg=0.):  
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

