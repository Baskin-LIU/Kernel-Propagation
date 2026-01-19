import torch
from .rho import *

class Neurons(torch.nn.Module):

    def __init__(self, n_in, n_neurons, bias=None, activation='linear', tau=20., lr_w=1e-2, 
                 dt=1., device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.n_in = n_in
        self.n_neurons = n_neurons #n_out
        
        self._init_tau(tau)
        self.lr_w = lr_w
        self.lr_b = lr_w*1e-1
        self.dt = dt

        dt_tau = self.dt/self.tau
        decay = 1 - dt_tau
        assert (decay>=0).all()
        
        self.register_buffer('dt_tau', dt_tau)
        self.register_buffer('decay', decay)

        self.next_layer = None
        self.previous_layer = None

        self.u_d = torch.zeros(1, n_neurons)
        self.u_bar = torch.zeros(1, n_neurons)
        self.r = torch.zeros(1, n_neurons) #output rate
        self.wTe = torch.zeros(1, n_neurons)
        self.r_in = torch.zeros(1, n_in)
        self.mismatch = 0.#torch.zeros(batch, n_neurons)
        self.epsilon = torch.zeros(1, n_neurons)
        
        self.activation = activation
        self.rho = rho(n_neurons)

        self.custom_init()
        self.LP = self.tau.max()>self.dt

    def E_trg(self, e_trg=0,):
        self.wTe = e_trg

    def custom_init(self, ):
        return
    
    def step(self,):
        return


    def set_next_layer(self, nxt):
        if self.next_layer is None:
            self.next_layer = [nxt]
        else:
            self.next_layer.append(nxt)

    def set_previous_layer(self, previous):
        if self.previous_layer is None:
            self.previous_layer = [previous]
        else:
            self.previous_layer.append(previous)
        
    def reset(self, ):
        self.u_bar = torch.zeros(1, self.n_neurons).to(self.device)
        self.wTe = torch.zeros(1, self.n_neurons).to(self.device)
        self.rho.reset()

    def _init_tau(self, tau):
        if type(tau) is tuple: #random initialize within the given range
            self.tau_min, self.tau_max = tau
            tau= torch.rand(self.n_neurons)*(self.tau_max - self.tau_min) + self.tau_min
        elif type(tau) is float or type(tau) is int: #scalar tau then apply same for all
            tau= torch.tensor(tau).repeat(self.n_neurons)
        elif type(tau) is list: #assigned tau
            tau= torch.tensor(tau)
        elif isinstance(tau, (np.ndarray, np.generic)):
            tau= torch.tensor(tau, dtype=torch.float32)
        elif isinstance(tau, torch.Tensor): #assigned tau
            tau= tau.clone()
        else:
            raise NotImplementedError

        self.register_buffer('tau', tau)
        assert len(self.tau) == self.n_neurons




class FwdNeurons(Neurons):
    
    def __init__(self, n_in, n_neurons, bias=None, activation='linear', tau=20., lr_w=1e-2, 
                 W_in=None, scale=0.8, dt=1., device="cpu"):
        super().__init__(n_in, n_neurons, bias, activation, tau, lr_w, dt, device)
        self.scale = scale
        self.rho = RHO[activation](n_neurons, scale=self.scale)
        self.weight_init(W_in, bias)

    def step(self, r_in, noise=0., **kwargs):
        self.r_in = r_in
        #update u and output
        self.u_d = self.r_in@self.W_in.T + self.bias
        self.u_bar = self.decay[None,:] * self.u_bar + self.dt_tau[None,:] * self.u_d
        self.r = self.rho(self.u_bar)

        return self.r, self.u_bar

    def step_bar(self, r_in, noise=0., **kwargs):
        self.r_bar = self.decay[None, :, None] * self.r_bar + self.dt_tau[None, :, None] * r_in[:, None, :]
        self.u_bar = (self.r_bar*self.W_in).sum(-1) + self.bias
        
        self.r = self.rho(self.u_bar)
    

    def prop(self, learn=True):
        self.epsilon = self.wTe * self.rho.d
        if learn and self.previous_layer is not None:
            self.previous_layer[0].wTe = self.epsilon @ self.W_in
        return 0, 0
        
        
    def learnW(self,):    
        dW_in = (self.epsilon.unsqueeze(dim=-1) * self.r_in[:,None,:]).mean(dim=0)
        self.W_in += dW_in * self.lr_w
        self.bias += self.epsilon.mean(0) * self.lr_b

    def backwards(self, ):
        self.W_in.grad -= (self.epsilon.unsqueeze(dim=-1) * self.r_in[:,None,:]).mean(dim=0)
        self.bias.grad -= self.epsilon.mean(0)

    def weight_init(self, W_in, bias):
        # init weight from last layer / input
        if W_in is not None:
            self.W_in = torch.nn.Parameter(W_in)
        else:
            # W_in = torch.randn(self.n_neurons, self.n_in)
            # W_in *= (2/(W_in**2).sum(dim=1, keepdim=True))**0.5
            W_in = torch.empty(self.n_neurons, self.n_in)
            torch.nn.init.kaiming_normal_(W_in, mode="fan_in", nonlinearity=self.activation)
            self.W_in = torch.nn.Parameter(W_in)

        if bias is not None:
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias = torch.nn.Parameter(torch.randn(self.n_neurons)/10.)

        self.W_in.grad = torch.zeros(self.n_neurons, self.n_in)
        self.bias.grad = torch.zeros(self.n_neurons)
        

        

class FwdInsNeurons(FwdNeurons):
    
    def custom_init(self,):
        self.tau = torch.zeros(self.n_neurons)
        

    def step(self, r_in, noise=0., **kwargs):
        self.r_in=r_in
        #update u and output
        self.u_d = self.r_in@self.W_in.T + self.bias
        self.u_bar = self.u_d
        self.r = self.rho(self.u_bar)

        return self.r, self.u_bar


class FwdGLENeurons(FwdNeurons):

    def custom_init(self,):
        self.mismatch = torch.zeros(self.n_neurons)
        tau_dt = (self.tau/self.dt)
        tau_dt[tau_dt==1] = 0.
        self.register_buffer('tau_dt', tau_dt)

    def prop(self, **kwargs):
        self.epsilon_past = self.epsilon    
        self.epsilon = self.wTe * self.rho.d
        self.mismatch = self.epsilon + self.tau_dt[None,:]*(self.epsilon-self.epsilon_past)
        if self.previous_layer != None:
            self.previous_layer[0].wTe = self.mismatch @ self.W_in
        return 0, 0

        
    def learnW(self,):
        dW_in = (self.mismatch.unsqueeze(dim=-1) * self.r_in[:,None,:]).mean(0)
        self.W_in += dW_in * self.lr_w
        self.bias += self.mismatch.mean(0) * self.lr_b

    def backwards(self, ):
        self.W_in.grad = -(self.mismatch.unsqueeze(dim=-1) * self.r_in[:,None,:]).mean(dim=0)
        self.bias.grad = -self.mismatch.mean(0)

    def reset(self,):
        super().reset()
        self.epsilon_past = torch.zeros(1, self.n_neurons)

        

class FwdRFNeurons(FwdNeurons):
    #TDOD optimize r_bar
    
    def custom_init(self, ):
        self.r_bar = torch.zeros(self.n_neurons, self.n_in) #r_in_bar
        
    def step(self, r_in, noise=0., **kwargs):
        self.step_bar(r_in, noise)
        return self.r, self.u_bar
    
    def prop(self, **kwargs):
        super().prop()
        self.r_bar = self.decay[None, :, None] * self.r_bar + self.dt_tau[None, :, None] * self.r_in[:, None, :]
        return 0, 0
        
    def learnW(self,):
        dW_in = (self.epsilon.unsqueeze(dim=-1) * self.r_bar).mean(0)
        self.W_in += dW_in * self.lr_w
        self.bias += self.epsilon.mean(0) * self.lr_b

    def backwards(self, ):
        self.W_in.grad = -(self.epsilon.unsqueeze(dim=-1) * self.r_bar).mean(dim=0)
        self.bias.grad = -self.epsilon.mean(0)

    def reset(self,):
        super().reset()
        self.r_bar = torch.zeros(1, self.n_neurons, self.n_in)


        
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

