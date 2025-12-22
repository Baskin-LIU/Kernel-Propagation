from rho import *
from FwdNeuron import *
import numpy as np
import torch
    
class LastFwdDENeurons(FwdNeurons):
    # last layer of LP neurons before inst output to save computation
        
    def custom_init(self, ):
        self.r_bar = torch.zeros(self.n_neurons, self.n_in) #r_in_bar
        
    def prop(self,):
        self.r_bar = self.decay[:, None] * self.r_bar + torch.outer(self.dt_tau, self.r_in)
        
        self.epsilon = self.rho.d * self.wTe
        
        self.B = self.epsilon[:, None].expand(-1, self.totalN)
        #self.B = torch.tile(self.epsilon[:, None],(1, self.totalN))
        if self.previous_layer != None:
            self.previous_layer.B = self.W_in.T @ (self.T * self.B)
        
        return 0, 0

    
    def learnW(self,):
        dW_in = self.epsilon.unsqueeze(dim=1) * self.r_bar
        self.W_in += dW_in * self.dt * self.lr_w
        self.b += self.epsilon * self.dt * self.lr_b


class FwdDENeurons(FwdNeurons):
    
    def custom_init(self, ):
        self.r_bar = torch.zeros(self.n_neurons, self.n_in) #r_in_bar

    def prop(self,):
        #Update DeepElig
        self.r_bar = self.decay[:, None] * self.r_bar + torch.outer(self.dt_tau, self.r_in) #n_neuron, n_in
        self.elig = self.decay_de[None, :, None] * self.elig + self.dt_tau_de[None, :, None] * (self.rho.d[:, None, None] * self.r_bar[:, None, :]) #n_neuron*n_exp*n_in
        self.elig_b = self.decay_de[None, :] * self.elig_b + self.dt_tau_de[None, :] * self.rho.d[:, None] #n_neuron*n_exp

        #Kernel Propagation
        if self.previous_layer != None:
            self.previous_layer.B = self.W_in.T @ (self.rho.d[:, None] * self.T * self.B)
        
        return 0, 0
        

    def learnW(self,):
        B = self.B[:,:self.downstream].clone()
        self.dW_in = (B.unsqueeze(-1) * self.elig).sum(axis=1)
        self.W_in += self.dW_in * self.dt * self.lr_w
        self.b += (B * self.elig_b).sum(axis=1) * self.dt * self.lr_b





class FwdDEInsNeurons(FwdDENeurons):
    # instantaneous layers at upper stream
    def prop(self,):
        #Update DeepElig
        self.elig = self.decay_de[None, :, None] * self.elig + self.dt_tau_de[None, :, None] * self.rho.d[:, None, None] * self.r[None, None, :] #n_neuron*n_exp*n_in  
        self.elig_b = self.decay_de[None, :] * self.elig_b + self.dt_tau_de[None, :] * self.rho.d[:, None] #n_neuron*n_exp

        #Kernel Propagation
        if self.previous_layer != None:
            self.previous_layer.B = self.W_in.T @ (self.rho.d[:, None] * self.T * self.B)
        
        return 0, 0


