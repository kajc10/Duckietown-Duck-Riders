import os
import gym
from gym_duckietown.simulator import Simulator
from gym_duckietown.envs.duckietown_env import DuckietownLF
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import *

from datetime import datetime
import logging
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.tune.registry import register_env
from ray.tune.logger import TBXLoggerCallback
from ray.rllib import _register_all
from ray.tune.trial import Trial

from wrappers import ResizeWrapper, NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, CropWrapper
import trainers
import argparse

# from Logger import TensorboardImageLogger

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=10, type=int)
parser.add_argument("--map_name", default='small_loop', type=str)
parser.add_argument("--max_steps", default=2000000, type=int)
parser.add_argument("--training_name", default='Training_results', type=str)
parser.add_argument("--num_cpus", default=None, type=int)
parser.add_argument("--num_gpus", default=None, type=int)
parser.add_argument("--mode", default='ppo', type=str)
args = parser.parse_args()

seed = args.seed
map_name = args.map_name
max_steps = args.max_steps
domain_rand = False
camera_width = 640
camera_height = 480
checkpoint_path = "./dump"
training_name = args.training_name
mode = args.mode

# https://docs.ray.io/en/latest/rllib-env.html
# simply passing env as an argument won't work..
'''
,,The gym registry is not compatible with Ray. 
  Instead, always use the registration flows 
  documented above to ensure Ray workers 
  can access the environment."
'''


def prepare_env(env_config):
    env = Simulator(
        seed=seed,
        map_name=map_name,
        max_steps=max_steps,
        domain_rand=domain_rand,
        camera_width=camera_width,
        camera_height=camera_height,
    )

    env = CropWrapper(env)
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ActionWrapper(env)  # max 80% speed
    env = DtRewardWrapper(env)

    return env


register_env("myenv", prepare_env)

# To explicitly stop or restart Ray, use the shutdown API.
ray.shutdown()

ray.init(
    num_cpus=args.num_cpus,
    num_gpus=args.num_gpus,
    include_dashboard=False,
    ignore_reinit_error=True,
    log_to_driver=False,
)

# print(ray.get_gpu_ids())

# Tune config
# Rlib uses built-in models if a custom one is not specified.
# (built-on ones can be modified via config as well)  https://docs.ray.io/en/latest/rllib-models.html
if mode == 'ddpg':
    analysis = trainers.ddpg(training_name, args.num_gpus, args.num_cpus-1, max_steps)
if mode == 'a2c':
    analysis = trainers.a2c(training_name, args.num_gpus, args.num_cpus - 1, max_steps)
else:
    analysis = trainers.ppo(training_name, args.num_gpus, args.num_cpus-1, max_steps)

print(
    "Best hyperparameters found:",
    analysis.best_config,
)

checkpoint_path = analysis.best_checkpoint

f = open("best_trial_checkpoint_path.txt", "w")
f.write(str(checkpoint_path))
f.close()
