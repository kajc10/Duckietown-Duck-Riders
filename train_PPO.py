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
from ray.tune.trial import Trial 

from wrappers import ResizeWrapper,NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper
import argparse

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=10, type=int)
parser.add_argument("--map_name", default='udem1', type=str)
parser.add_argument("--max_steps", default=3, type=int)
parser.add_argument("--training_name", default='Training_results', type=str)
args = parser.parse_args()

seed = args.seed
map_name      =  args.map_name
max_steps     =  args.max_steps
domain_rand   =  False
camera_width  =  640
camera_height =  480
checkpoint_path = "./dump"
training_name = args.training_name


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
	
	env = ResizeWrapper(env)
	env = NormalizeWrapper(env)
	#env = ImgWrapper(env) 
	env = ActionWrapper(env)  #max 80% speed
	env = DtRewardWrapper(env)

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

#Tune config
#Rlib uses built-in models if a custom one is not specified.
#(built-on ones can be modified via config as well)  https://docs.ray.io/en/latest/rllib-models.html
parameter_search_analysis = ray.tune.run(
	PPOTrainer,
	name=training_name,
	#Trainer (algorithm) config
	config={
		"env": "myenv",         
		"framework": "torch",

		#Model config
		"model": {
			"fcnet_activation": "relu"
		}
	},
	stop={'timesteps_total': max_steps},
	num_samples=1,
	metric="timesteps_total",
	mode="min",
       checkpoint_at_end=True,
       local_dir="./dump",
       keep_checkpoints_num=1,
       checkpoint_score_attr="episode_reward_mean",
       checkpoint_freq=1,
)

print(
	"Best hyperparameters found:",
	parameter_search_analysis.best_config,
)

checkpoint_path = parameter_search_analysis.best_checkpoint

f = open("best_trial_checkpoint_path.txt", "w")
f.write(str(checkpoint_path))
f.close()
