# train_dqn.py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from iot_env import IoTEnv
import os

os.makedirs('results', exist_ok=True)
env = make_vec_env(lambda: IoTEnv(n_devices=6), n_envs=1)
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=20000)
model.save('results/dqn_iot')
print('saved results/dqn_iot')
