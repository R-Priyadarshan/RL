# evaluate.py
import numpy as np
from iot_env import IoTEnv
from stable_baselines3 import DQN

env = IoTEnv(n_devices=6)
model = DQN.load('results/dqn_iot')
obs = env.reset()
total = 0
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(int(action))
    total += reward
print('Total reward (200 steps):', total)
