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


#https://docs.ray.io/en/latest/rllib-env.html
#simply passing env as an argument won't work..
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
		max_steps = max_steps,
		domain_rand=domain_rand,
		camera_width=camera_width,
		camera_height=camera_height,
	)
	
	#env = ResizeWrapper(env)
	#env = NormalizeWrapper(env)
	#env = ImgWrapper(env)     #160x120x3 into 3x160x120
	#env = ActionWrapper(env)  #max 80% speed
	#env = DtRewardWrapper(env)
	#env = gym.make("CartPole-v0")
	
	#env = gym.make("CartPole-v0")
	return env


register_env("myenv", prepare_env)


# To explicitly stop or restart Ray, use the shutdown API.
ray.shutdown()

ray.init(
	num_cpus=3,
	include_dashboard=False,
	ignore_reinit_error=True,
	log_to_driver=False,
)

parameter_search_analysis = ray.tune.run(
	PPOTrainer,
	config={"env": "myenv",
		"framework": "torch"
	},
	stop={'timesteps_total': max_steps},
	num_samples=1,
	metric="timesteps_total",
	mode="min",
)

print(
	"Best hyperparameters found:",
	parameter_search_analysis.best_config,
)



