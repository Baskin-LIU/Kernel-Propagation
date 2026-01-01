from rho import *
from FwdNeuron import *
import numpy as np
import torch
    
class LastFwdDENeurons(FwdNeurons):
    # last layer of LP neurons before inst output to save computation
        
    def step(self, r_in, noise=0., **kwargs):
        self.step_bar(r_in, noise)
        return self.r, self.u_bar
        
    def prop(self,):
        #self.r_bar = self.decay[None, :, None] * self.r_bar + self.dt_tau[None, :, None] * self.r_in[:, None, :]
        self.epsilon = self.rho.d * self.wTe
        if self.previous_layer != None:
            self.previous_layer.K = (self.P * self.epsilon[:, None, :]) @ self.W_in
        
        return 0, 0

    
    def learnW(self,):
        dW_in = (self.epsilon.unsqueeze(dim=-1) * self.r_bar).mean(0)
        self.W_in += dW_in * self.dt * self.lr_w
        self.bias += self.epsilon.mean(0) * self.dt * self.lr_b
        

    def backwards(self, ):
        dW_in = (self.epsilon.unsqueeze(dim=-1) * self.r_bar).mean(0)
        self.W_in.grad = -dW_in.clone()
        self.bias.grad = -self.epsilon.mean(0) * self.dt

    def reset(self,):
        self.u_bar = torch.zeros(1, self.n_neurons)
        self.r_bar = torch.zeros(1, self.n_neurons, self.n_in)
        self.rho.reset()
        


class FwdDENeurons(FwdNeurons):

    def step(self, r_in, noise=0., **kwargs):
        self.step_bar(r_in, noise)
        return self.r, self.u_bar

    def prop(self,):
        # Update r_bar (batch, n_neuron, n_in)
        #self.r_bar = self.decay[None, :, None] * self.r_bar + self.dt_tau[None, :, None] * self.r_in[:, None, :]
        # Update eligibility trace (batch, n_exp, n_neuron, n_in)
        self.elig = self.decay_de[None,:,None,None] * self.elig + self.dt_tau_de[None,:,None,None] * (self.rho.d[:,None,:,None] * self.r_bar[:,None,:,:])
        # Bias eligibility (batch, n_exp, n_neuron)
        self.elig_b = self.decay_de[None,:,None] * self.elig_b + self.dt_tau_de[None, :, None] * self.rho.d[:, None, :]
        # Kernel propagation
        if self.previous_layer is not None:  #(batch,n_exp,n_neuron)*(n_neuron,n_in) -> (batch,n_exp,n_in)
            self.previous_layer.K = (self.rho.d[:, None, :] * self.P * self.K) @ self.W_in
        
        return 0, 0
    
    def learnW(self,):
        K = self.K[:,:self.downstream].clone()
        dW_in = (K.unsqueeze(-1)*self.elig).sum(1).mean(0)*self.dt
        self.W_in += dW_in * self.lr_w
        self.bias += (K * self.elig_b).sum(1).mean(0) * self.dt * self.lr_b

    def backwards(self, ):
        K = self.K[:,:self.downstream].clone()
        dW_in = (K.unsqueeze(-1)*self.elig).sum(axis=1).mean(dim=0)*self.dt
        self.W_in.grad = -dW_in.clone()
        self.bias.grad = -(K * self.elig_b).sum(axis=1).mean(dim=0) * self.dt

    def reset(self,):
        #batch = self.batch if batch is None else batch
        self.u_bar = torch.zeros(1, self.n_neurons)
        self.r_bar = torch.zeros(1, self.n_neurons, self.n_in)
        self.elig = torch.zeros(1,self.downstream,self.n_neurons,self.n_in)
        self.elig_b = torch.zeros(1, self.downstream, self.n_neurons)
        self.rho.reset()


##TODO Reduced version. r_bar in hidden neurons with same tau can be shared. Which can also be used for forwarding W r_bar.
class FwdDENeuronsReduced(FwdDENeurons):

    def custom_init(self, ):
        self.repeat_tau_matrix=None #TODO
    
    def step(self, ):
        self.r_in = r_in
        self.r_bar_uniq = self.decay_uniq[None, :, None] * self.r_bar + self.dt_tau_uniq[None, :, None] * self.r_in[:, None, :]
        self.r_bar = torch.tile(self.r_bar_uniq, (1, self.repeat_tau, 1)) #switch to a matrix, 1 for that tau
        #update u and output
        self.u_bar = (self.r_bar*self.W_in.T).sum(-1) + self.bias
        self.r = self.rho(self.u_bar)

        return self.r, self.u_bar

    def prop(self,):
        # Update reduced eligibility trace (batch, n_exp, uniq_tau, n_in)
        self.elig_uniq = self.decay_de[None,:,None,None] * self.elig_uniq + self.dt_tau_de[None,:,None,None] * self.r_bar_uniq[:,None,:,:]
        # Tile to eligibility trace (batch, n_exp, n_neuron, n_in)
        self.elig = self.elig_uniq
        # Bias eligibility (batch, n_exp, n_neuron)
        self.elig_b_uniq = self.decay_de * self.elig_b_uniq + self.dt_tau_de
        self.elig_b =  self.elig_b_uniq[None, :, None]
        # Kernel propagation
        if self.previous_layer is not None:  #(batch,n_exp,n_neuron)*(n_neuron,n_in) -> (batch,n_exp,n_in)
            self.previous_layer.K = (self.rho.d[:, None, :] * self.P * self.K) @ self.W_in
        
        return 0, 0



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


