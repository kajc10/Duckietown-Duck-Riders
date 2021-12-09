import gym
from gym import spaces
import numpy as np
from PIL import Image
from gym_duckietown.simulator import Simulator
from gym_duckietown.simulator import NotInLane


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
        self.prev_pos = None

    def reward(self, reward):
        my_reward = 0
        
        #Gettin current position and stored for later
        pos = self.cur_pos
        angle = self.cur_angle
        prev_pos = self.prev_pos
        self.prev_pos = pos
        
        if prev_pos is None:
            return my_reward
        
        #Compute travelled distance
        curve_point, curve_tangent = self.closest_curve_point(pos, angle)
        prev_curve_point, prev_curve_tangent = self.closest_curve_point(prev_pos, angle)
        travelled_dist = np.linalg.norm(curve_point - prev_curve_point)
        
        #Compute lane position relative to the center of the rigth lane
        lane_pos = self.get_lane_pos2(pos, self.cur_angle)
        
        if lane_pos is NotInLane:
                print("Not In Lane")
                return my_reward
        
        #Compute reward
        #if the agent leaves the right lane: center of the road + agent width/2 =  -0.105 + 0.13/2 = -0.04
        if lane_pos.dist < -0.04:
        	print("Not In Good Lane")
        	return my_reward
        
        if np.dot(curve_tangent, curve_point - prev_curve_point) < 0:
            print("Moving backward")
            return my_reward
	
	#maximum travelled distance = maximum speed*timestep = 1.2*1/30 = 0.04
        travelled_dist_reward = np.interp(travelled_dist, (0, 0.04), (0,1))
        lane_center_dist_reward = np.interp(np.abs(lane_pos.dist), (0, 0.04), (1, 0))
        lane_center_angle_reward = np.interp(np.abs(lane_pos.angle_deg), (0, 30), (1,0))
	
        W1 = 1
        W2 = 1
        W3 = 1
        my_reward = (W1*travelled_dist_reward + W2*lane_center_dist_reward + W3*lane_center_angle_reward)/3 
       
        print("Travelled_dist: ", travelled_dist_reward)
        print("Center_dist: ", lane_center_dist_reward)
        print("Angle: ", lane_center_angle_reward)
        return my_reward
