import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
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
    'rho_scale': 0.6,
    'LP_size': (32, 50, 60), 
    'Ins_size': (60, 30), 
    'activation': 'tanh', 
    "reducedNonlinear": False,
    'Tau0': (5, 40, 4), 
    'Tau1': np.array([5 , 5 , 10 , 10, 10]), 
    'Tau2': np.array([11, 11, 11, 24, 24]),
}



class WindCartPoleEnv(CartPoleEnv):

    def __init__(self, wind=0.0, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.wind = wind

    def step(self, action):

        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag

        # add wind disturbance
        force += self.wind

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass

        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        terminated = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        reward = 1.0

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


class CartPoleCustomize(gym.Wrapper):
    def __init__(self, env=None, display=False, obs='velocity'):
        # step size 20 ms
        if env is None:
            if display:
                env = gym.make("CartPole-v1", render_mode="rgb_array")
            else:
                env = gym.make("CartPole-v1")
        super().__init__(env)

        env = self.env.unwrapped
        # Angle at which to fail the episode
        self.angle_threshold = 18 #12
        self.x_threshold = 3.6 #2.4
        env.theta_threshold_radians = self.angle_threshold * 2 * np.pi / 360
        env.x_threshold = self.x_threshold
        
        self.angle_punish_threshold = 6*2*np.pi/360
        self.x_punish_threshold = 1.5

        if obs == 'velocity':
            self.obs_mask = [1, 3]
        elif obs == 'position':
            self.obs_mask = [0, 2]
        else:
            self.obs_mask = [0, 1, 2, 3]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.set_masspole(0.1)
        return torch.tensor(obs[self.obs_mask])[None, :], info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward = reward - 5*np.maximum(0, np.abs(obs[2]) - self.angle_punish_threshold) - np.maximum(0, np.abs(obs[0]) - self.x_punish_threshold)
        obs = torch.tensor(obs[self.obs_mask])[None, :]

        return obs, reward, done, truncated, info

    def set_masspole(self, m):
        env = self.env.unwrapped
        env.masspole = m
        env.total_mass = env.masspole + env.masscart
        env.polemass_length = env.masspole * env.length

    def set_wind(self, wind):
        env = self.env.unwrapped
        env.wind = wind



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
                scale=model_config["rho_scale"],
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


def episode(env, actor, critic, critic_target, optimizer_actor, optimizer_critic):
    # Run one episode
    observation, _ = env.reset()
    actor.reset()
    critic.reset()
    critic_target.reset()

    terminated, truncated = False, False
    total_reward=0
    gamma=0.9

    with torch.no_grad():
        while not terminated and not truncated:
            # Render the environment
            #env.render()
            # Take a random action
            u, _ = actor.step(observation)
            p = torch.softmax(u, dim=1)
            action = np.random.choice(n_actions, p=p[0].cpu().numpy())
            value,_ = critic.step(observation)
            # Step the environment
            observation_next, reward, terminated, truncated,_ = env.step(action)
            value_next,_ = critic_target.step(observation)

            delta = reward + gamma*value_next - value

            optimizer_actor.zero_grad(set_to_none=False)
            optimizer_critic.zero_grad(set_to_none=False)
            
            actor.prop(F.one_hot(torch.tensor(action), num_classes=n_actions)-p)
            actor.backwardsRL(delta.item(), gamma)
            critic.prop(1.)
            critic.backwardsRL(delta.item(), gamma)

            optimizer_actor.step()
            optimizer_critic.step()

            total_reward+=reward

        hard_update(critic_target, critic)
        #soft_update(critic_target, critic, tau=0.01)

    return total_reward

@torch.no_grad()
def soft_update(target, source, tau = 0.01):
    for p_t, p in zip(target.parameters(), source.parameters()):
        p_t.mul_(1.0 - tau)
        p_t.add_(tau * p)

def hard_update(target, source):
    target.load_state_dict(source.state_dict())


