import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from Neurons.FwdNeuron import *
from Neurons.DeepEligNeuron import *
from Neurons.RLNeuron import *
from Network import *

### DEFAULT Config ####
default_general_config = {
    'seed': 0, 
    'dt': 5, 
    'device': 'cpu', 
    'short_training_run': False,
    }

default_data_config = {
    'task': 'cartpole', 
    'n_observation': 2,
    'n_action': 2,
    }

default_train_config = {
    'num_epochs': 100, 
    'learning_rate_actor': 1e-4, 
    'learning_rate_critic': 1e-3, 
    'batch_size': 1, 
    'num_workers': 4, 
    'num_prefetch_batch': 2,
    }

default_model_config = {
    'n_in': 2, 
    'n_out': 3, 
    'num_LP_layers': 3, 
    'num_Ins_layers': 1, 
    'LP_size': (32, 50, 60), 
    'Ins_size': (60, 30), 
    'activation': 'tanh', 
    "reducedNonlinear": False,
    'Tau0': (5, 40, 4), 
    'Tau1': np.array([5 , 5 , 10 , 10, 10]), 
    'Tau2': np.array([11, 11, 11, 24, 24]),
}

class CartPoleCustomize(gym.Wrapper):
    def __init__(self, display=False):
        # step size 20 ms
        if display:
            env = gym.make("CartPole-v1", render_mode="rgb_array")
        else:
            env = gym.make("CartPole-v1")
        super().__init__(env)
        self.obs_mask = [1, 3]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.set_masspole(0.1)
        return torch.tensor(obs[self.obs_mask])[None, :], info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward = reward - np.abs(obs[2]).sum() - 0.08*np.abs(obs[0]).sum()
        obs = torch.tensor(obs[self.obs_mask])[None, :]

        return obs, reward, done, truncated, info

    def set_masspole(self, m):
        env = self.env.unwrapped
        env.masspole = m
        env.total_mass = env.masspole + env.masscart
        env.polemass_length = env.masspole * env.length



def buildRLNet(model_config, general_config):
    device = general_config["device"]
    LP_size = list(model_config['LP_size'])
    Ins_size = list(model_config['Ins_size'])
    dt = general_config['dt']
    tau = []
    tau_min, tau_max, repeat = model_config['Tau0']
    tau0 = np.logspace(np.log10(tau_min), np.log10(tau_max), LP_size[0]//repeat, dtype=np.float32)
    tau.append(np.repeat(tau0[:, None], repeat))
    for i in range(model_config['num_LP_layers']-1):
        tau_uniq = model_config['Tau%d'%(i+1)]
        tau.append(np.repeat(tau_uniq[:, None],
                                          LP_size[i+1]//tau_uniq.shape[0]))
    layers = torch.nn.ModuleList()
    prev_n=model_config['n_in']
    
    layers.append(
        FwdDERLNeurons(
            n_in=prev_n,
            n_neurons=LP_size[0],
            tau=tau[0], 
            activation=model_config["activation"], 
            dt=dt, 
            scale=1.,
            device=device,
        )
    )
    prev_n=LP_size[0]

    for i in range(model_config['num_LP_layers']-2):
        layers.append(
            FwdDERLNeurons(
                n_in=prev_n,
                n_neurons=LP_size[i+1],
                tau=tau[i+1], 
                activation=model_config["activation"], 
                dt=dt, 
                scale=0.6,
                device=device,
            )
        )
        prev_n=LP_size[i+1]

    layers.append(
        LastFwdDERLNeurons(
            n_in=prev_n, 
            n_neurons=LP_size[model_config['num_LP_layers']-1], 
            tau=tau[model_config['num_LP_layers']-1], 
            activation=model_config["activation"], 
            dt=dt, 
            scale=1.,
            device=device,
            )
    )
    prev_n=LP_size[model_config['num_LP_layers']-1]
    for i in range(model_config['num_Ins_layers']):
        layers.append(
            FwdInsRLNeurons(
                n_in=prev_n,
                n_neurons=Ins_size[i],
                activation=model_config["activation"], 
                scale=1.0,
                dt=dt,
                device=device,
            )
        )
        prev_n=Ins_size[i]
                
    layers.append(
        FwdInsRLNeurons(
            n_in=prev_n, 
            n_neurons=model_config['n_out'],
            activation='linear', 
            dt=dt, 
            scale=1.0,
            device=device,
            )
    )
    
    return DEFwdNetwork(layers=layers, dt=dt, device=device)


