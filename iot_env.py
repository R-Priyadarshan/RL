# iot_env.py
import gym
from gym import spaces
import numpy as np

class IoTEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, n_devices=5):
        super().__init__()
        self.n = n_devices
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n*2,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n)
        self.reset()

    def reset(self):
        self.energy = np.ones(self.n)
        self.queues = np.random.randint(0,5,size=self.n).astype(np.float32)
        return self._obs()

    def _obs(self):
        return np.concatenate([self.energy, self.queues/10.0]).astype(np.float32)

    def step(self, action):
        reward = 0.0
        if self.queues[action] > 0 and self.energy[action] > 0.1:
            self.queues[action] -= 1
            self.energy[action] -= 0.1
            reward += 1.0
        # small decay
        self.energy = np.maximum(0, self.energy - 0.01)
        done = False
        info = {}
        return self._obs(), reward, done, info

    def render(self, mode='human'):
        print('energy', self.energy, 'queues', self.queues)
