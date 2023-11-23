from stable_baselines3.common.env_checker import check_env
from snakeenv import SnakeEnv
from stable_baselines3 import PPO
import os
from gym import RewardWrapper

ALGORTIHM="PPO"
models_dir = "models/" + ALGORTIHM
log_dir = "logs"
TIMESTEPS = 1000000

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
env = SnakeEnv()

model = PPO.load(f'{models_dir}/{ALGORTIHM}_{TIMESTEPS}_2')

obs,info=env.reset()
for i in range(2000):
	action,_state = model.predict(obs,deterministic=True)
	obs,reward,done,info,_ = env.step(action)
	if done:
		obs,info = env.reset()
	env.close()

check_env(env)
