import gymnasium as gym
import numpy as np

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
    'learning_rate': 1e-4, 
    'batch_size': 40, 
    'num_workers': 4, 
    'num_prefetch_batch': 2,
    }

default_model_config = {
    'n_in': 2, 
    'n_out': 3, 
    'num_LP_layers': 2, 
    'num_Ins_layers': 1, 
    'LP_size': (40, 40), 
    'Ins_size': (60, 60,), 
    'activation': 'tanh', 
    "reducedNonlinear": False,
    'Tau0': (1, 4), 
    'Tau1': np.array([0.6, 1.1 , 1.1 , 2.1 , 2.1 , 2.1]), #np.array([0.6, 4. , 4. , 4. , 9. , 9.]),#
    'Tau2': np.array([0.5, 1.2, 1.2, 1.2, 3.1, 3.1]),#np.array([0.5, 5.1, 5.1, 5.1, 8.1, 8.1]),#
    }

class CartPoleWithWind(gym.Env):
    def __init__(self, wind_strength=0.0):
        self.env = gym.make("CartPole-v1")
        self.wind_strength = wind_strength

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        # Take a step in the original environment
        state, reward, done, info = self.env.step(action)

        # Modify the state with wind effect
        self.apply_wind(state)

        return state, reward, done, info

    def apply_wind(self, state):
        # Assuming state[2] represents the cart's position (x)
        # Apply wind effect to the cart's position
        wind_effect = np.random.uniform(-self.wind_strength, self.wind_strength)
        state[0] += wind_effect  # state[0] is the cart position

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()



class CustomBipedalWalker(gym.envs.box2d.BipedalWalker):
    def __init__(self, wind_force=0.0):
        super().__init__()
        self.wind_force = wind_force

    def step(self, action):
        # Call the original step function
        state, reward, done, info = super().step(action)
        
        # Modify the agent's state based on wind force
        if self.wind_force != 0.0:
            # Example: apply a constant force in the x direction
            state[0] += self.wind_force  # Assuming index 0 is x position
        return state, reward, done, info

class CustomLunarLander(gym.envs.box2d.LunarLander):
    def __init__(self, wind_force=0.0):
        super().__init__()
        self.wind_force = wind_force

    def step(self, action):
        # Call the original step function
        state, reward, done, info = super().step(action)
        
        # Modify the agent's state based on wind force
        if self.wind_force != 0.0:
            # Example: apply a constant force in the x direction
            state[0] += self.wind_force  # Assuming index 0 is x position
        return state, reward, done, info


