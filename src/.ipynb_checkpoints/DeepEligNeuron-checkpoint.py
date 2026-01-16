from rho import *
from FwdNeuron import *
import numpy as np
import torch
    
class LastFwdDENeurons(FwdNeurons):
    # last layer of LP neurons before inst output to save computation
        
    def step(self, r_in, noise=0., **kwargs):
        self.step_bar(r_in, noise)
        return self.r, self.u_bar
        
    def prop(self, learn=True):
        self.epsilon = self.rho.d * self.wTe
        if learn and self.previous_layer is not None:
            self.previous_layer[0].K = (self.P * self.epsilon[:, None, :]) @ self.W_in
        
        return 0, 0
    
    def learnW(self,):
        dW_in = (self.epsilon.unsqueeze(dim=-1) * self.r_bar).mean(0)
        self.W_in += dW_in * self.lr_w
        self.bias += self.epsilon.mean(0) * self.lr_b
        
    def backwards(self, ):
        self.W_in.grad -= (self.epsilon.unsqueeze(dim=-1) * self.r_bar).mean(0)
        self.bias.grad -= self.epsilon.mean(0)

    def reset(self,):
        super().reset()
        self.r_bar = torch.zeros(1, self.n_neurons, self.n_in).to(self.device)
        


class FwdDENeurons(FwdNeurons):

    def step(self, r_in, noise=0., **kwargs):
        self.step_bar(r_in, noise)
        return self.r, self.u_bar

    def prop(self, learn=True):
        # Update eligibility trace (batch, n_exp, n_neuron, n_in)
        self.elig = self.decay_de[None,:,None,None] * self.elig + self.dt_tau_de[None,:,None,None] * (self.rho.d[:,None,:,None] * self.r_bar[:,None,:,:])
        # Bias eligibility (batch, n_exp, n_neuron)
        self.elig_b = self.decay_de[None,:,None] * self.elig_b + self.dt_tau_de[None, :, None] * self.rho.d[:, None, :]
        # Kernel propagation
        if learn and self.previous_layer is not None:  
            #(batch,n_exp,n_neuron)*(n_neuron,n_in) -> (batch,n_exp,n_in)
            self.previous_layer[0].K = (self.rho.d[:, None, :] * self.P * self.K) @ self.W_in
        
        return 0, 0
    
    def learnW(self,):
        K = self.K[:,:self.downstream].clone()
        dW_in = (K.unsqueeze(-1)*self.elig).sum(1).mean(0)
        self.W_in += dW_in * self.lr_w
        self.bias += (K * self.elig_b).sum(1).mean(0) * self.lr_b

    def backwards(self, ):
        K = self.K[:,:self.downstream].clone()
        self.W_in.grad = -(K.unsqueeze(-1)*self.elig).sum(axis=1).mean(dim=0)
        self.bias.grad = -(K * self.elig_b).sum(axis=1).mean(dim=0)

    def reset(self,):
        super().reset()
        self.r_bar = torch.zeros(1, self.n_neurons, self.n_in).to(self.device)
        self.elig = torch.zeros(1,self.downstream,self.n_neurons,self.n_in).to(self.device)
        self.elig_b = torch.zeros(1, self.downstream, self.n_neurons).to(self.device)


##TODO Reduced version. r_bar in hidden neurons with same tau can be shared. Which can also be used for forwarding W r_bar.
class FwdDENeuronsReduced(FwdDENeurons):

    def custom_init(self, ):
        _, repeat_tau = torch.unique(self.tau, sorted=False, return_counts=True)
        self.register_buffer("repeat_tau", repeat_tau)

    
    def step(self, r_in, noise=0., **kwargs):
        self.r_in = r_in
        self.r_bar_uniq = self.decay_uniq[None, :, None] * self.r_bar_uniq + self.dt_tau_uniq[None, :, None] * self.r_in[:, None, :]
        self.r_bar = torch.repeat_interleave(self.r_bar_uniq, self.repeat_tau, dim=1)
        #update u and output
        self.u_bar = (self.r_bar*self.W_in).sum(-1) + self.bias
        self.r = self.rho(self.u_bar)

        return self.r, self.u_bar

    def prop(self, learn=True):
        # Update reduced eligibility trace (batch, n_exp, uniq_tau, n_in)
        self.elig_uniq = self.decay_de[None,:,None,None] * self.elig_uniq + self.dt_tau_de[None,:,None,None] * self.r_bar_uniq[:,None,:,:]
        # Tile to eligibility trace (batch, n_exp, n_neuron, n_in)
        self.elig = torch.repeat_interleave(self.elig_uniq, self.repeat_tau, dim=2)
        # Bias eligibility (batch, n_exp, n_neuron)
        self.elig_b = self.decay_de[None, :, None] * self.elig_b + self.dt_tau_de[None, :, None]
        # Kernel propagation
        if learn and self.previous_layer is not None:  #(batch,n_exp,n_neuron)*(n_neuron,n_in) -> (batch,n_exp,n_in)
            self.previous_layer[0].K = (self.P * self.K) @ self.W_in
        
        return 0, 0

    def reset(self,):
        super().reset()
        self.r_bar_uniq = torch.zeros(1, self.n_tau, self.n_in).to(self.device)
        self.elig_uniq = torch.zeros(1,self.downstream,self.n_tau,self.n_in).to(self.device)
        self.elig_b = torch.zeros(1, self.downstream, 1).to(self.device)



class FwdDEInsNeurons(FwdDENeurons):
    # instantaneous layers at upper stream
    def prop(self,):
        #Update DeepElig
        self.elig = self.decay_de[None, :, None] * self.elig + self.dt_tau_de[None, :, None] * self.rho.d[:, None, None] * self.r[None, None, :] #n_neuron*n_exp*n_in  
        self.elig_b = self.decay_de[None, :] * self.elig_b + self.dt_tau_de[None, :] * self.rho.d[:, None] #n_neuron*n_exp

        #Kernel Propagation
        if self.previous_layer != None:
            self.previous_layer[0].B = self.W_in.T @ (self.rho.d[:, None] * self.T * self.B)
        
        return 0, 0


