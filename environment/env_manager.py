import gym
from config import Config

def get_env(cfg):
    env_type = cfg.env_type
    if env_type == 'gym':
        env = gym.make(cfg.env_name)
    if env_type == 'custom':
        print('[info] You choose custom environment')

    return env