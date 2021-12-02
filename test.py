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

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


seed = 10
map_name      =  'udem1'
max_steps     =  3
domain_rand   =  False
camera_width  =  640
camera_height =  480
checkpoint_path = './dump'
training_name = '2021_12_01_1'


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

model = PPOTrainer(config=trainer_config, env="myenv")
model.restore('./dump/PPO_learning/PPO_myenv_526f1_00000_0_2021-12-02_22-17-58/checkpoint_000001/checkpoint-1')



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
for step in range(30):
	action, _= model.compute_single_action(obs)
	print('ACTION computed: ',action)
	observation, reward, done, info = env.step(env.action_space.sample())
	#observation, reward, done, info = env.step(action)
	print('NEW: ',observation, reward, done, info)
	
	env.render()
	time.sleep(0.5)
env.close()


