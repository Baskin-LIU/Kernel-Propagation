import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from FwdNeuron import *
from DeepEligNeuron import *
from Network import *
from inputFuc import *
from utils import *
from plotting import *

### DEFAULT Config ####
default_general_config = {
    'seed': 0, 
    'dt': 0.5, 
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
    'learning_rate': 5e-4, 
    'batch_size': 40, 
    'num_workers': 4, 
    'num_prefetch_batch': 2,
    }

default_model_config = {
    'n_in': 2, 
    'n_out': 3, 
    'num_LP_layers': 3, 
    'num_Ins_layers': 1, 
    'LP_size': (60, 60, 60), 
    'Ins_size': (60, ), 
    'activation': 'tanh', 
    "reducedNonlinear": False,
    'Tau0': (1, 6), 
    'Tau1': np.array([2. , 2. , 2. , 3., 3.]), 
    'Tau2': np.array([1.1, 1.1, 1.1, 2.4, 2.4]),
}

class CartPoleCustomize(gym.Wrapper):
    def __init__(self, switch_step=200, new_masspole=0.2):
        env = gym.make("CartPole-v1")
        super().__init__(env)
        self.switch_step = switch_step
        self.new_masspole = new_masspole
        self.obs_mask = [0, 2]
        self.t = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.t = 0
        self._set_masspole(0.1)
        return torch.tensor(obs[self.obs_mask])[None, :], info

    def step(self, action):
        self.t += 1
        if self.t == self.switch_step:
            self._set_masspole(self.new_masspole)
        obs, reward, done, truncated, info = self.env.step(action)

        obs = torch.tensor(obs[self.obs_mask])[None, :]
        reward = reward - torch.abs(obs[:, 1]).sum().item()

        return obs, reward, done, truncated, info

    def _set_masspole(self, m):
        return
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
    tau_min, tau_max = model_config['Tau0']
    tau0 = np.logspace(np.log10(tau_min), np.log10(tau_max), LP_size[0]//3, dtype=np.float32)
    tau.append(np.repeat(tau0[:, None], 3))
    for i in range(model_config['num_LP_layers']-1):
        tau_uniq = model_config['Tau%d'%(i+1)]
        tau.append(np.repeat(tau_uniq[:, None],
                                          LP_size[i+1]//tau_uniq.shape[0]))
    layers = torch.nn.ModuleList()
    prev_n=model_config['n_in']
    
    layers.append(
        FwdDENeurons(
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
        if model_config["reducedNonlinear"]:   
            layers.append(
                FwdDENeuronsReduced(
                    n_in=prev_n,
                    n_neurons=LP_size[i+1],
                    tau=tau[i+1], 
                    activation="linear", 
                    dt=dt, 
                    device=device,
                )
            )
            prev_n=LP_size[i+1]
        else:
            layers.append(
                FwdDENeurons(
                    n_in=prev_n,
                    n_neurons=LP_size[i+1],
                    tau=tau[i+1], 
                    activation=model_config["activation"], 
                    dt=dt, 
                    scale=0.4,
                    device=device,
                )
            )
            prev_n=LP_size[i+1]
    layers.append(
        LastFwdDENeurons(
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
            FwdInsNeurons(
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
        FwdInsNeurons(
            n_in=prev_n, 
            n_neurons=model_config['n_out'],
            activation='linear', 
            dt=dt, 
            scale=1.0,
            device=device,
            )
    )
    
    return DEFwdNetwork(layers=layers, device=device)


