import numpy as np
import torch
import numpy.random as random
from collections import deque
import torch.nn.functional as F

class sin_function():
    def __init__(self, n, period, bias= 0., magnitude = 1., phase = 0., dt=1., rand_init=True, dtype = torch.float32):
        self.n = n
        self.dt = torch.tensor(dt)

        if rand_init: # random initialize within range
            self.period_min, self.period_max = period
            self.bias_min, self.bias_max = bias
            self.magnitude_min, self.magnitude_max = magnitude
            self.phase_min, self.phase_max = phase
            
            self.period = torch.rand(self.n)*(self.period_max - self.period_min) + self.period_min
            self.bias = torch.rand(self.n)*(self.bias_max - self.bias_min) + self.bias_min
            self.magnitude = torch.rand(self.n)*(self.magnitude_max - self.magnitude_min) + self.magnitude_min
            self.phase = torch.rand(self.n)*(self.phase_max - self.phase_min) + self.phase_min
            self.frequency = 1/self.period

        else:
            self.period = period
            self.bias = bias
            self.magnitude = magnitude
            self.phase = phase
            self.frequency = 1/self.period

    def __call__(self, t):
        sin = self.magnitude*torch.sin(t*self.dt*self.frequency*2*np.pi + self.phase) + self.bias
        return sin

class OU_process():
    def __init__(self, size, tau, mu, sigma, dt=1., rand_init=True, dtype = torch.float32):
        self.n = size
        self.dt = torch.tensor(dt)
        self.sqrt_dt = torch.sqrt(self.dt)
        self.x = torch.zeros(self.n)
        
        if rand_init: # random initialize within range
            self.tau_min, self.tau_max = tau
            self.mu_min, self.mu_max = mu
            self.sigma_min, self.sigma_max = sigma
            
            self.tau = torch.rand(self.n)*(self.tau_max - self.tau_min) + self.tau_min
            self.mu = torch.rand(self.n)*(self.mu_max - self.mu_min) + self.mu_min
            self.sigma = torch.rand(self.n)*(self.sigma_max - self.sigma_min) + self.sigma_min
        
        else:
            self.tau = tau
            self.mu = mu
            self.sigma = sigma

        self.theta = 1/self.tau
        

    def __call__(self, t=None):
        dx = self.theta * (self.mu - self.x) * self.dt + self.sigma * self.sqrt_dt * torch.randn(self.n)
        self.x = self.x + dx
        return self.x


class mul_sin_function():
    def __init__(self, n, n_sin, period, bias= 0., magnitude = 1., phase = 0., dt=1., rand_init=True, dtype = torch.float32):
        self.n = n
        self.dt = torch.tensor(dt)
        self.n_sin = n_sin

        self.sin_func = sin_function(n*n_sin, period, bias, magnitude, phase, dt, rand_init, dtype)

    def __call__(self, t):
        sin = self.sin_func(t)
        imposed = sin.reshape(-1, self.n_sin).sum(axis=1)
        return imposed


class low_pass_OU():
    def __init__(self, size, lp_tau, tau, mu, sigma, dt=1., rand_init=True, dtype = torch.float32):
        self.n = size
        self.dt = torch.tensor(dt)
        self.x = torch.zeros(self.n)
    
        self.OU = OU_process(size, tau, mu, sigma, dt, rand_init, dtype)

        if rand_init: # random initialize within range
            self.tau_min, self.tau_max = lp_tau        
            self.lp_tau = torch.rand(self.n)*(self.tau_max - self.tau_min) + self.tau_min
        
        else:
            self.lp = lp_tau

        self.dt_tau = self.dt/self.lp_tau

    def __call__(self, t=None):
        dx = self.OU()
        self.x += self.dt_tau*(-self.x + dx)
        return self.x

'''
OU_input = low_pass_OU(size=4, lp_tau=(20, 50), tau=(10, 100), mu=(0, 0), sigma=(0.1, 0.2), dt = 0.1)
duration = 20
x = torch.zeros(1000, 4)
for i in range(1000):
    x[i] = OU_input()
for i in range(4):
    plt.plot(x[:, i])
'''

class gratingS():
    def __init__(self, n=5, frequency=10, dt=1., dtype = torch.float32):
        self.n = n
        self.nn = n*n
        self.dt = dt
        self.frequency = frequency
        self.period = 1000//(frequency*dt)

    def __call__(self, t, direction):
        self.grid=torch.zeros(self.n, self.n)
        bar_loc = np.random.randint(0, self.n)
        if direction:
            self.grid[bar_loc] = 1/self.dt
        else:
            self.grid[:, bar_loc] = 1/self.dt
        
        self.activity=torch.zeros(int(t/self.dt), self.nn)
        self.activity[np.arange(0, int(t/self.dt), self.period)]+= torch.flatten(self.grid)
        
        return self.activity


class gratingR():
    def __init__(self, n=5, frequency=10, dt=1., dtype = torch.float32):
        self.n = n
        self.nn = n*n
        self.dt = dt

    def __call__(self, t, direction):
        self.grid=torch.zeros(self.n, self.n)
        bar_loc = np.random.randint(0, self.n)
        if direction:
            self.grid[bar_loc] = 1
        else:
            self.grid[:, bar_loc] = 1
        
        self.activity = torch.flatten(self.grid)
        
        return self.activity


class movegrating():
    def __init__(self, n=5, frequency=10, dtype = torch.float32):
        self.n = n
        self.nn = n*n
        self.dt = torch.tensor(dt)
        self.frequency = 1/self.period

    def __call__(self, t, direction):
        self.grid=torch.zeros(n, n)
        self.t
        return sin




class ParityBits():
    def __init__(self, N, seed=None):
        """
        N = size of the parity window
        seed = optional RNG seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        self.N = N
        self.window = deque(maxlen=N)

    def step(self):
        """
        Generate one new bit, update memory window,
        return (bit, parity).
        """
        bit = random.randint(0, 2)

        # update sliding window
        self.window.append(bit)

        # compute parity over last N bits
        # parity is XOR of bits -> 0 if even number of 1s, 1 if odd
        parity = 0
        for b in self.window:
            parity ^= b

        return bit, parity

class StepParity():
    def __init__(self, N=2, bit_time=5, dt=0.1, dtype=torch.float32):
        self.N = N
        self.parity = ParityBits(N)
        
        self.bit_time = bit_time #ms
        self.dt = torch.tensor(dt)
        
        self.label = torch.zeros(2)
        self.next_label = 0
        self.bit = 0
        self.t = 0

    def __call__(self, ):
        self.t += self.dt
        if self.t >= self.bit_time:
            self.t = 0
            self.label *= 0   
            self.bit, self.next_label = self.parity.step()
            self.label[self.next_label] = 1.

        return torch.tensor([self.bit-0.5]), self.label

class SumBits():
    def __init__(self, N, seed=None):
        """
        N = size of the parity window
        seed = optional RNG seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        self.N = N
        self.window = deque(maxlen=N)

    def step(self):
        """
        Generate one new bit, update memory window,
        return (bit, parity).
        """
        bit = random.randint(0, 2)

        # update sliding window
        self.window.append(bit)

        sum = np.array(self.window).sum()

        return bit, sum

class StepSum():
    def __init__(self, N=2, bit_time=5, dt=0.1, dtype=torch.float32):
        self.N = N
        self.sum = SumBits(N)
        
        self.bit_time = bit_time #ms
        self.dt = torch.tensor(dt)
        
        self.label = 0
        self.bit = 0
        self.t = 0

    def __call__(self, ):
        self.t += self.dt
        if self.t >= self.bit_time:
            self.t = 0
            self.bit, self.label = self.sum.step()

        return torch.tensor([self.bit-0.5]), torch.tensor([self.label/self.N], dtype=torch.float32)

class StepSumBi():
    def __init__(self, N=2, bit_time=5, dt=0.1, dtype=torch.float32):
        self.N = N
        self.sum = SumBits(N)
        
        self.bit_time = bit_time #ms
        self.dt = torch.tensor(dt)
        
        self.label = 0
        self.bit = 0
        self.t = 0
        self.lenLabel = self.N.bit_length()
        self.binary_label = torch.zeros(self.lenLabel)

    def __call__(self, ):
        self.t += self.dt
        if self.t >= self.bit_time:
            self.t = 0
            self.bit, self.label = self.sum.step()
            self.binary_label = torch.tensor(list(map(int, f"{self.label:0{self.lenLabel}b}")))

        return torch.tensor([self.bit-0.5]), self.binary_label


class NBack:
    def __init__(
        self,
        N=4,
        n_class=10,
        show_time=10.0,
        dt=1.0,
        max_N=20,
        dtype=torch.float32,
        device="cpu",
    ):
        assert N <= max_N

        self.N = N
        self.n_class = n_class
        self.show_time = show_time
        self.dt = dt
        self.max_N = max_N
        self.dtype = dtype
        self.device = device

        self.show_steps = int(show_time / dt)

        # memory stores class indices
        self.memory = np.zeros(max_N, dtype=np.int64)

        self.t = 0
        self.current_symbol = None

    def __call__(self):
        # At symbol boundaries, sample a new class
        if self.t % self.show_steps == 0:
            new_symbol = random.randint(self.n_class)

            # shift memory left, push new symbol
            self.memory[:-1] = self.memory[1:]
            self.memory[-1] = new_symbol

            self.current_symbol = new_symbol

        # input: one-hot of current symbol
        x = F.one_hot(
            torch.tensor(self.current_symbol, device=self.device),
            num_classes=self.n_class,
        ).to(self.dtype)

        # target: symbol N steps back
        target = torch.tensor(
            self.memory[-(self.N + 1)],
            dtype=torch.int64,
            device=self.device,
        )

        self.t += 1
        return x, target

    def change_N(self, new_N):
        assert 0 <= new_N <= self.max_N
        self.N = new_N

    def reset(self):
        self.memory[:] = 0
        self.t = 0
        self.current_symbol = None







        