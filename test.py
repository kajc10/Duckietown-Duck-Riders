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

from wrappers import ResizeWrapper,NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper
from ray.tune.integration.wandb import WandbLogger
import time
import sys

import argparse

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=10, type=int)
parser.add_argument("--map_name", default='udem1', type=str)
parser.add_argument("--max_steps", default=500, type=int)
parser.add_argument("--training_name", default='Training_results', type=str)
args = parser.parse_args()

seed = args.seed
map_name = args.map_name
max_steps     =  args.max_steps
domain_rand   =  False
camera_width  =  640
camera_height =  480
checkpoint_path = './dump'
training_name = args.training_name

def prepare_env(env_config):
	env = Simulator(
		seed=seed,
		map_name=map_name,
		max_steps = max_steps,
		domain_rand=domain_rand,
		camera_width=camera_width,
		camera_height=camera_height,
	)
	
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
	}
}

f = open("best_trial_checkpoint_path.txt", "r")
chechpoint_path = f.readline()
f.close()

model = PPOTrainer(config=trainer_config, env="myenv")
model.restore = checkpoint_path

env = Simulator(
	seed=seed,
	map_name=map_name,
	max_steps = max_steps,
	domain_rand=domain_rand,
	camera_width=camera_width,
	camera_height=camera_height,
)

env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ActionWrapper(env)  #max 80% speed
env = DtRewardWrapper(env)

# predicing actions and stepping with them
obs = env.reset()

for step in range(500):
	action,_,_= model.compute_single_action(observation=obs,full_fetch=True)
	print('ACTION computed: ',action)
	observation, reward, done, info = env.step(action)
	print('NEW: ',observation, reward, done, info)
	print('REWARD: ', reward)
	
	env.render()
	time.sleep(0.25)
env.close()
