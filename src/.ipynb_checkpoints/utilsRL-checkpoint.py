import gymnasium as gym
import numpy as np

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


