class RecNeurons(Neurons):
    
    def __init__(self, n_in, n_neurons, n_a=None, u_rest=0., W_in=None, W_r=None, tau=20., 
                 inh=0.1, num_P=1, lr_w = 1e-2, lr_p = 1e-1, beta=0.1,
                 activation='linear', afunc='self', P_bias=False, dt=1., device="cpu",):
        super().__init__(n_in, n_neurons, n_a, u_rest, tau, lr_w, lr_p, beta, dt, activation, afunc, device)
        
        self.inh = inh
        self.rho = RHO[activation](n_neurons)
        # init weight from last layer / input
        self.W_in = torch.nn.Parameter(torch.randn(n_neurons, n_in), requires_grad=False
                                      ) if W_in is None else torch.nn.Parameter(W_in, requires_grad=False)

        diagonal = torch.diag(-torch.rand(n_neurons))
        Q = torch.randn(n_neurons, n_neurons)
        W = Q @ diagonal @ torch.inverse(Q) - self.inh
        self.W_r = torch.nn.Parameter(W, requires_grad=False
                                     ) if W_r is None else torch.nn.Parameter(W_r, requires_grad=False)

        # init error predition matrix
        self.a = torch.zeros(self.n_neurons, self.n_a)
        if P_bias:
            self.a[:, -1] = 2.
        self.a_last = self.a
        self.num_P = num_P
        self.P = torch.nn.Parameter(torch.zeros(self.num_P, self.n_neurons, self.n_a), requires_grad=False)
        self.i = 0 ###task_index
        

    def step(self, r_in, noise=0., **kwargs):
        self.update_a_last()
        self.update_a(self, **kwargs)
        self.mismatch = (self.P[self.i] * self.a).sum(dim=1)

        self.r_in=r_in
        #update u and output
        self.u_d = (self.W_in * self.r_in).sum(dim=1) + (self.W_r * self.r).sum(dim=1) + self.mismatch

        if self.previous_layer != None:
            self.previous_layer.wTe = self.W_in.T @ self.mismatch

        self.u_bar = self.u_bar + self.dt_tau*(self.u_rest - self.u_bar + self.u_d) + noise
        self.r = self.rho(self.u_bar)

        return self.r, self.u_bar
    

    def learnP(self, e_trg=0, learning=True):
            
        self.epsilon = (self.W_r.T @ self.mismatch + self.wTe + self.beta * e_trg) * self.rho.d

        a_hat = (1.-self.tau_dt) * self.a + self.tau_dt * self.a_last
        Pa_hat = (self.P[self.i] * a_hat).sum(dim=1)

        err = self.epsilon - Pa_hat
        dP = err.unsqueeze(dim=1)*self.a_last
        
        self.P[self.i] += self.lr_p * dP * self.dt

        return err, dP


    def learnW(self):
        dW_in = self.mismatch.unsqueeze(dim=1) * self.r_in
        self.W_in += dW_in * self.dt * self.lr_norm()

        dW_r = self.mismatch.unsqueeze(dim=1) * self.r
        self.W_r += dW_r * self.dt * self.lr_norm()

class RecGLENeurons(Neurons):
    
    def __init__(self, n_in, n_neurons, n_a=None, u_rest=0., W_in=None, W_r=None, tau=20., lr_w = 1e-2, lr_p = 1e-1, beta=0.1,
                 activation='linear', afunc='inp', dt=1., device="cpu",):
        super().__init__(n_in, n_neurons, n_a, u_rest, tau, lr_w, lr_p, beta, dt, activation, afunc, device)

        if W_r is None:
            diagonal = torch.diag(-torch.rand(n_neurons))
            Q = torch.randn(n_neurons, n_neurons)
            W = Q @ diagonal @ torch.inverse(Q) - self.inh
            self.W_r = torch.nn.Parameter(W, requires_grad=False)
        else:
            self.W_r = torch.nn.Parameter(W_r, requires_grad=False)

        self.rho = RHO[activation](n_neurons)
        # init weight from last layer / input
        self.W_in = torch.nn.Parameter(torch.randn(n_neurons, n_in), requires_grad=False
                                      ) if W_in is None else torch.nn.Parameter(W_in, requires_grad=False)        

    def step(self, r_in, noise=0.):
        self.r_in=r_in
        #update u and output
        self.u_d = (self.W_in * self.r_in).sum(dim=1) + (self.W_r * self.r).sum(dim=1) + self.mismatch

        if self.previous_layer != None:
            self.previous_layer.wTe = self.W_in.T @ self.mismatch

        self.u_bar = self.u_bar + self.dt_tau*(self.u_rest - self.u_bar + self.u_d) + noise
        self.r = self.rho(self.u_bar)

        return self.r, self.u_bar
    

    def learnP(self, e_trg=0, learning=True):
        self.epsilon_past = self.epsilon    
        self.epsilon = (self.W_r.T @ self.mismatch + self.wTe + self.beta * e_trg) * self.rho.d

        self.mismatch = self.epsilon + self.tau_dt.squeeze(1)*(self.epsilon-self.epsilon_past)

        self.mismatch *= self.lr_norm()
        return 0, 0

    def learnW(self):
        dW_in = self.mismatch.unsqueeze(dim=1) * self.r_in
        self.W_in += dW_in * self.dt * self.lr_w

        dW_r = self.mismatch.unsqueeze(dim=1) * self.r
        self.W_r += dW_r * self.dt * self.lr_w


class RecLENeurons(Neurons):
    
    def __init__(self, n_in, n_neurons, n_a=None, u_rest=0., W_in=None, W_r=None, tau=20., lr_w = 1e-2, lr_p = 1e-1, beta=0.1,
                 activation='linear', afunc='inp', dt=1., device="cpu", inh=0.1):
        super().__init__(n_in, n_neurons, n_a, u_rest, tau, lr_w, lr_p, beta, dt, activation, afunc, device)
        self.inh=inh/self.n_neurons
        if W_r is None:
            diagonal = torch.diag(-torch.rand(n_neurons))
            Q = torch.randn(n_neurons, n_neurons)
            W = Q @ diagonal @ torch.inverse(Q) - self.inh
    
            self.W_r = torch.nn.Parameter(W, requires_grad=False)
        else:
            self.W_r = torch.nn.Parameter(W_r, requires_grad=False)

        self.rho = RHO[activation](n_neurons)
        # init weight from last layer / input
        self.W_in = torch.nn.Parameter(torch.randn(n_neurons, n_in), requires_grad=False
                                      ) if W_in is None else torch.nn.Parameter(W_in, requires_grad=False)
        

    def step(self, r_in, noise=0.):
        self.r_in=r_in
        #update u and output
        self.u_d = (self.W_in * self.r_in).sum(dim=1) + (self.W_r * self.r).sum(dim=1) + self.mismatch

        if self.previous_layer != None:
            self.previous_layer.wTe = self.W_in.T @ self.mismatch

        self.u_bar = self.u_bar + self.dt_tau*(self.u_rest - self.u_bar + self.u_d) + noise
        self.r = self.rho(self.u_bar)

        return self.r, self.u_bar
    
    def learnP(self, e_trg=0, learning=True):  
        self.epsilon = (self.W_r.T @ self.mismatch + self.wTe + self.beta * e_trg) * self.rho.d
        self.mismatch = self.epsilon
        return 0, 0

    def learnW(self):
        dW_in = self.mismatch.unsqueeze(dim=1) * self.r_in
        self.W_in += dW_in * self.dt*self.lr_w

        dW_r = self.mismatch.unsqueeze(dim=1) * self.r
        self.W_r += dW_r * self.dt*self.lr_w
        

class RecEPNeurons(Neurons):
    
    def __init__(self, n_in, n_neurons, n_a=None, u_rest=0., W_in=None, W_r=None, tau=20., lr_w = 1e-2, lr_p = 1e-1, beta=0.1,
                 activation='linear', afunc='inp', P_bias=False, dt=1., device="cpu", inh=0.1):
        super().__init__(n_in, n_neurons, n_a, u_rest, tau, lr_w, lr_p, beta, dt, activation, afunc, device)
        self.inh=inh/self.n_neurons
        if W_r is None:
            diagonal = torch.diag(-torch.rand(n_neurons))
            Q = torch.randn(n_neurons, n_neurons)
            W = Q @ diagonal @ torch.inverse(Q) - self.inh
            W /= 0.6*self.n_neurons*torch.abs(W).mean()
    
            self.W_r = torch.nn.Parameter(W, requires_grad=False)
        else:
            self.W_r = torch.nn.Parameter(W_r, requires_grad=False)

        self.rho = RHO[activation](n_neurons)
        # init weight from last layer / input
        self.W_in = torch.nn.Parameter(torch.randn(n_neurons, n_in), requires_grad=False
                                      ) if W_in is None else torch.nn.Parameter(W_in, requires_grad=False)
        self.W_in /= self.n_in*torch.abs(self.W_in).mean()
        self.elig = torch.zeros(self.n_neurons, self.n_in) #r_in_bar
        self.elig_r = torch.zeros(self.n_neurons, self.n_neurons) #r_bar
        

    def step(self, r_in, noise=0.):
        self.r_in=r_in
    
        #update u and output
        self.u_d = (self.W_in * self.r_in).sum(dim=1) + (self.W_r * self.r).sum(dim=1) + self.b

        if self.previous_layer != None:
            self.previous_layer.wTe = self.W_in.T @ self.mismatch

        self.u_bar = self.u_bar + self.dt_tau*(self.u_rest - self.u_bar + self.u_d) + noise
        self.r = self.rho(self.u_bar)
        
        self.elig = (1-self.dt_tau.unsqueeze(1))*self.elig + torch.outer(self.dt_tau, self.r_in)
        self.elig_r = (1-self.dt_tau.unsqueeze(1))*self.elig_r + torch.outer(self.dt_tau, self.r)

        return self.r, self.u_bar
    

    def learnP(self, e_trg=0):  
        self.epsilon = (self.wTe + self.beta * e_trg) * self.rho.d
        self.mismatch = self.epsilon
        return 0, 0

    def learnW(self):
        dW_in = self.mismatch.unsqueeze(dim=1) * self.elig
        self.W_in += dW_in * self.dt * self.lr_w  #* self.lr_norm() #

        dW_r = self.mismatch.unsqueeze(dim=1) * self.elig_r
        self.W_r += dW_r * self.dt *self.lr_w #* self.lr_norm() #

        if self.bias:
            self.b += self.mismatch * self.dt * self.lr_w * 0.5


class RecLEEPNeurons(RecEPNeurons):
    
    def learnP(self, e_trg=0, learning=True):
        self.epsilon = (self.wTe + self.beta * e_trg) * self.rho.d
        self.mismatch = self.epsilon + (self.W_r.T@self.epsilon*self.dt_tau)*self.rho.d # t+2
        return 0, 0


class RecLEEP1Neurons(RecEPNeurons):

        
    def learnP(self, e_trg=0):
        self.epsilon = (self.wTe + self.beta * e_trg) * self.rho.d
        self.mismatch = self.epsilon + (self.W_r.T@self.epsilon*self.dt_tau)*self.rho.d # t+2
        self.mismatch = self.epsilon + (self.W_r.T@self.mismatch*self.dt_tau)*self.rho.d #t+1

        return 0, 0
    
class RecLEEP2Neurons(RecEPNeurons):

    def learnP(self, e_trg=0, learning=True):
        self.epsilon = (self.wTe + self.beta * e_trg) * self.rho.d
        self.mismatch = self.epsilon + (self.W_r.T@self.epsilon*self.dt_tau)*self.rho.d # t+3
        self.mismatch = self.epsilon + (self.W_r.T@self.mismatch*self.dt_tau)*self.rho.d #t+2
        self.mismatch = self.epsilon + (self.W_r.T@self.mismatch*self.dt_tau)*self.rho.d #t+1
        
        return 0, 0

class RecLEEP3Neurons(RecEPNeurons):
    def experimental_init(self,):
        self.epsilon_past = torch.zeros(self.n_neurons)
        self.epsilon_dot = torch.zeros(self.n_neurons)
    
    def learnP(self, e_trg=0):
        #self.epsilon_past = self.epsilon    
        self.epsilon = (self.wTe + self.beta * e_trg) * self.rho.d

        self.mismatch = self.epsilon + self.W_r.T @ self.epsilon*self.dt_tau * self.rho.d

        # self.epsilon_past = self.epsilon    
        # self.epsilon = (self.W_r.T @ self.mismatch + self.wTe + self.beta * e_trg) * self.rho.d

        # self.mismatch = (self.epsilon+(self.epsilon-self.epsilon_past))*self.dt_tau
        
        # self.epsilon = (self.W_r.T @ (self.mismatch*self.dt_tau+) 
        #                 + self.wTe + self.beta * e_trg) * self.rho.d
        # self.mismatch = self.epsilon
        return 0, 0