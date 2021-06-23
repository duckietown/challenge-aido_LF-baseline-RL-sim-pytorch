import cv2
import gym
import numpy as np
from gym import spaces
from PIL import Image

__all__ = [
    "ResizeWrapper",
    "NormalizeWrapper",
    "ImgWrapper",
    "DtRewardWrapper",
    "SpeedActionWrapper",
    "SteeringToWheelVelWrapper",
    "DTPytorchWrapper",
    "FakeWrap",
]


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype,
        )
        self.shape = shape

    def observation(self, observation):
        return np.array(Image.fromarray(observation).resize(self.shape[0:2]))


class NormalizeWrapper(gym.ObservationWrapper):
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


# This is needed because at max speed the duckie can't turn anymore
class SpeedActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_

    def reverse_action(self, action):
        raise NotImplementedError()


class SteeringToWheelVelWrapper(gym.ActionWrapper):
    """
    Converts policy that was trained with [velocity|heading] actions to
    [wheelvel_left|wheelvel_right] to comply with AIDO evaluation format
    """

    def __init__(self, env, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, wheel_dist=0.102):
        gym.ActionWrapper.__init__(self, env)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

        self.wheel_dist = wheel_dist

    def action(self, action):
        vel, angle = action

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * self.wheel_dist) / self.radius
        omega_l = (vel - 0.5 * angle * self.wheel_dist) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels

    def reverse_action(self, action):
        raise NotImplementedError()


class FakeWrap:
    def __init__(self):
        self.env = None
        self.action_space = None

        self.camera_width = 640
        self.camera_height = 480
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.camera_height, self.camera_width, 3), dtype=np.uint8
        )
        self.reward_range = None
        self.metadata = None


class DTPytorchWrapper:
    def __init__(self, shape=(64, 64, 3)):
        self.shape = shape
        self.transposed_shape = (shape[2], shape[0], shape[1])

    def preprocess(self, obs):
        # from PIL import Image
        # return np.array(Image.fromarray(obs).resize(self.shape[0:2])).transpose(2, 0, 1)

        obs = cv2.resize(obs, self.shape[0:2])
        # NOTICE: OpenCV changes the order of the channels !!!
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        return obs.transpose(2, 0, 1)
