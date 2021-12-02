import gym
from gym import spaces
import numpy as np
from PIL import Image
from gym_duckietown.simulator import Simulator


'''
These wrappers are from the original gym-duckietown repo with slight modifications
'''

class ResizeWrapper(gym.ObservationWrapper):
    '''
    Original observation images have shape: (480,640,3)
    New shape: ,,No default configuration for obs shape [X, Y, 3], you must specify `conv_filters` manually as a model option. 
    		  Default configurations are only available for  inputs of shape [42, 42, K] and [84, 84, K]. 
    		  You may alternatively want to use a custom model or preprocessor. "
    For simplicity the default (84,84,K) shape was chosen...
    '''
    def __init__(self, env=None, shape=(84, 84, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            self.shape,
            dtype=self.observation_space.dtype
        )
        
    #https://github.com/raghakot/keras-vis/issues/209
    def observation(self, observation):
        return np.array(Image.fromarray(obj=observation).resize(size=self.shape[:2]))

	

class NormalizeWrapper(gym.ObservationWrapper):
    '''
    Observations to 0-1
    '''
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)
            

class ActionWrapper(gym.ActionWrapper):
    '''
    Needed because at max speed the duckie can't turn anymore
    '''
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_
        
class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    '''
    
    '''
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward