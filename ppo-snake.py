from stable_baselines3.common.env_checker import check_env
from snakeenv import SnakeEnv
from stable_baselines3 import PPO
import os
from gym import RewardWrapper

ALGORTIHM="PPO"
models_dir = "models/" + ALGORTIHM
log_dir = "logs"


if not os.path.exists(models_dir):
    os.makedirs(models_dir)


if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
env = SnakeEnv()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 1000000

model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps=False, tb_log_name=ALGORTIHM, progress_bar = True)
model.save(f"{models_dir}/{ALGORTIHM}_{TIMESTEPS}_2")
