import torch
from .FwdNeuron import *
from .DeepEligNeuron import *

class FwdInsRLNeurons(FwdInsNeurons):

    def backwardsRL(self, delta, gamma, labd):
        self.eligRL = gamma * labd * self.eligRL - (self.epsilon.unsqueeze(dim=-1) * self.r_in[:,None,:]).mean(dim=0)
        self.eligbiasRL = gamma * labd * self.eligbiasRL - self.epsilon.mean(0)
        self.W_in.grad += self.eligRL * delta
        self.bias.grad += self.eligbiasRL * delta

    def reset(self, ):
        super().reset()
        self.eligRL = 0.
        self.eligbiasRL = 0.


class LastFwdDERLNeurons(LastFwdDENeurons):
    
    def backwardsRL(self, delta, gamma, labd):
        self.eligRL = 0
        self.eligbiasRL = 0
        self.eligRL = gamma * labd * self.eligRL - (self.epsilon.unsqueeze(dim=-1) * self.r_bar).mean(0)
        self.eligbiasRL = gamma * labd * self.eligbiasRL - self.epsilon.mean(0)
        self.W_in.grad += self.eligRL * delta
        self.bias.grad += self.eligbiasRL * delta

    def reset(self, ):
        super().reset()
        self.eligRL = 0.
        self.eligbiasRL = 0.


class FwdDERLNeurons(FwdDENeuronsV2):

    def backwardsRL(self, delta, gamma, labd):
        K = self.K[:,:self.downstream].clone()
        self.eligRL = gamma * labd * self.eligRL - (K.unsqueeze(-1)*self.elig).sum(axis=1).mean(dim=0)
        self.eligbiasRL = gamma * labd * self.eligbiasRL - (K * self.elig_b).sum(axis=1).mean(dim=0)
        self.W_in.grad += self.eligRL * delta
        self.bias.grad += self.eligbiasRL * delta

    def reset(self, ):
        super().reset()
        self.eligRL = 0.
        self.eligbiasRL = 0.