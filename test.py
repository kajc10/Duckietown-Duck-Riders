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
from ray.tune.logger import TBXLogger
from ray.rllib import _register_all

from wrappers import ResizeWrapper,NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, CropWrapper, GreyscaleWrapper
from ray.tune.integration.wandb import WandbLogger
import time
import sys

import argparse

from load_checkpoint import load_checkpoint_path

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=10, type=int)
parser.add_argument("--map_name", default='small_loop', type=str)
parser.add_argument("--max_steps", default=500, type=int)
parser.add_argument("--training_name", default='Training_results', type=str)
parser.add_argument("--model_name", default='', type=str)
parser.add_argument("--load", default='./dump', type=str)
args = parser.parse_args()

seed = args.seed
map_name = args.map_name
max_steps     =  args.max_steps
domain_rand   =  False
camera_width  =  640
camera_height =  480
checkpoint_path = './dump'
training_name = args.training_name
model_name = args.model_name

def prepare_env(env_config):
	env = Simulator(
		seed=seed,
		map_name=map_name,
		max_steps = max_steps,
		domain_rand=domain_rand,
		camera_width=camera_width,
		camera_height=camera_height,
	)
	
	env = CropWrapper(env)
	env = GrayscaleWrapper(env)
	env = ResizeWrapper(env)
	env = NormalizeWrapper(env)
	env = ActionWrapper(env)  #max 80% speed
	env = DtRewardWrapper(env)

	return env


register_env("myenv", prepare_env)

# Loading trained model
trainer_config = {
	"env": "myenv",         
	"framework": "torch",

	#Model config
		"model": {
		"fcnet_activation": "relu"
	},
	"env_config": {
			"accepted_start_angle_deg": 5,
		},
}

#f = open("best_trial_checkpoint_path.txt", "r")
#chechpoint_path = f.readline()
#f.close()

model = PPOTrainer(config=trainer_config, env="myenv")

path = checkpoint_path + '/' + args.training_name

if model_name:
	path = path+ '/' + args.model_name

checkpoint_path = load_checkpoint_path(path)
model.restore(checkpoint_path)

env = Simulator(
	seed=seed,
	map_name=map_name,
	max_steps = max_steps,
	domain_rand=domain_rand,
	camera_width=camera_width,
	camera_height=camera_height,
)

env = CropWrapper(env)
env = GrayscaleWrapper(env)
env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ActionWrapper(env)  #max 80% speed
env = DtRewardWrapper(env)

# predicing actions and stepping with them
obs = env.reset()

for i in range(5):
        obs = env.reset()
        env.render()
        done = False
        while not done:
            action = model.compute_action(obs, explore=False)
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.01)
env.close()
