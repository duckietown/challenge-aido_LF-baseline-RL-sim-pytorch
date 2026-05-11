"""Observation, reward, and action wrappers used by learning scripts."""

from collections.abc import Sequence
from typing import Any, SupportsFloat, cast

import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces

WHEEL_BASELINE = 0.102
WHEEL_RADIUS = 0.0318
MOTOR_CONSTANT = 27
MOTOR_GAIN = 1
MOTOR_TRIM = 0
PWM_LIMIT = 1
DEFAULT_MOTION_BLUR_WEIGHTS = (0.8, 0.15, 0.04, 0.01)
VELOCITY_LIMIT = 1
STEERING_LIMIT = 1


def _resize_observation_nearest(
    observation: np.ndarray,
    target_shape: Sequence[int],
) -> np.ndarray:
    target_height, target_width, _ = target_shape
    source_height, source_width, _ = observation.shape
    if (source_height, source_width) == (target_height, target_width):
        return observation
    y_indices = np.linspace(0, source_height - 1, target_height)
    x_indices = np.linspace(0, source_width - 1, target_width)
    rounded_y_indices = np.rint(y_indices)
    rounded_x_indices = np.rint(x_indices)
    y_indices = rounded_y_indices.astype(int)
    x_indices = rounded_x_indices.astype(int)
    return observation[y_indices][:, x_indices]


class MotionBlurWrapper(gym.Wrapper):
    """Approximate motion blur by averaging recent frames."""

    _frame_skip: int

    def __init__(
        self,
        environment: Env,
        frame_skip: int = 3,
    ) -> None:
        """Store the number of extra frames used for motion blur."""
        super().__init__(environment)
        self._frame_skip = frame_skip

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the environment and average the recent observations."""
        observations: list[np.ndarray] = []
        reward: SupportsFloat = 0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        for _ in range(self._frame_skip + 1):
            observation, reward, terminated, truncated, info = self.env.step(
                action,
            )
            converted_observation = np.asarray(observation)
            observations.append(converted_observation)
            if terminated or truncated:
                break
        weights = DEFAULT_MOTION_BLUR_WEIGHTS[: len(observations)]
        observation = np.average(observations, axis=0, weights=weights)
        observation = observation.astype(observations[-1].dtype, copy=False)
        return observation, float(reward), terminated, truncated, info


class ResizeWrapper(gym.ObservationWrapper):
    """Resize observations without relying on deprecated SciPy APIs."""

    observation_space: spaces.Box

    def __init__(
        self,
        environment: Env,
        shape: tuple[int, int, int] = (120, 160, 3),
    ) -> None:
        """Initialize the resized observation space."""
        super().__init__(environment)
        observation_space = cast("spaces.Box", self.observation_space)
        self.observation_space = spaces.Box(
            observation_space.low[0, 0, 0],
            observation_space.high[0, 0, 0],
            shape,
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Return the resized observation."""
        return _resize_observation_nearest(
            observation,
            self.observation_space.shape,
        )


class NormalizeWrapper(gym.ObservationWrapper):
    """Normalize wrapper for the DuckiematrixDB21J environment."""

    _observation_space_high: float
    _observation_space_low: float
    observation_space: spaces.Box

    def __init__(self, environment: Env) -> None:
        """Initialize the normalized observation space bounds."""
        super().__init__(environment)
        observation_space = cast("spaces.Box", self.observation_space)
        self._observation_space_low = float(observation_space.low[0, 0, 0])
        self._observation_space_high = float(
            observation_space.high[0, 0, 0],
        )
        self.observation_space = spaces.Box(
            0,
            1,
            observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalize the observation into the `[0, 1]` range."""
        observation = np.asarray(observation, dtype=np.float32)
        observation_space_low = self._observation_space_low
        observation_space_high = self._observation_space_high
        if observation_space_low == 0 and observation_space_high == 1:
            return observation
        return (observation - observation_space_low) / (
            observation_space_high - observation_space_low
        )


class ImageWrapper(gym.ObservationWrapper):
    """Image wrapper for the DuckiematrixDB21J environment."""

    observation_space: spaces.Box

    def __init__(self, environment: Env) -> None:
        """Transpose the observation space from HWC to CHW."""
        super().__init__(environment)
        observation_space = cast("spaces.Box", self.observation_space)
        observation_space_shape = observation_space.shape
        self.observation_space = spaces.Box(
            observation_space.low[0, 0, 0],
            observation_space.high[0, 0, 0],
            (
                observation_space_shape[2],
                observation_space_shape[0],
                observation_space_shape[1],
            ),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Return the channel-first image observation."""
        return observation.transpose(2, 0, 1)


class RewardWrapper(gym.RewardWrapper):
    """Reward wrapper for the DuckiematrixDB21J environment."""

    _REWARD = -1000

    def __init__(self, environment: Env) -> None:
        """Initialize the reward wrapper."""
        super().__init__(environment)

    def reward(self, reward: SupportsFloat) -> float:
        """Adjust the environment reward for RL training."""
        reward = float(reward)
        if reward == self._REWARD:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4
        return reward


class ActionWrapper(gym.ActionWrapper):
    """Convert policy actions [velocity, steering] to wheel PWM."""

    action_space: spaces.Box

    def __init__(self, environment: Env) -> None:
        """Initialize the action bounds for the RL policy."""
        super().__init__(environment)
        self.action_space = spaces.Box(
            low=np.array([0, -STEERING_LIMIT], dtype=np.float32),
            high=np.array([VELOCITY_LIMIT, STEERING_LIMIT], dtype=np.float32),
            dtype=np.float32,
        )

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attributes to the wrapped environment."""
        return getattr(self.env, name)

    def action(self, action: np.ndarray) -> np.ndarray:
        """Convert velocity and steering into left/right wheel PWM."""
        velocity, steering = np.asarray(action, dtype=np.float32)
        velocity *= 0.8
        k_r_inverse = (MOTOR_GAIN + MOTOR_TRIM) / MOTOR_CONSTANT
        k_l_inverse = (MOTOR_GAIN - MOTOR_TRIM) / MOTOR_CONSTANT
        omega_right = (
            velocity + 0.5 * steering * WHEEL_BASELINE
        ) / WHEEL_RADIUS
        omega_left = (
            velocity - 0.5 * steering * WHEEL_BASELINE
        ) / WHEEL_RADIUS
        pwm_right = np.clip(omega_right * k_r_inverse, -PWM_LIMIT, PWM_LIMIT)
        pwm_left = np.clip(omega_left * k_l_inverse, -PWM_LIMIT, PWM_LIMIT)
        return np.array([pwm_left, pwm_right], dtype=np.float32)


ImgWrapper = ImageWrapper
DtRewardWrapper = RewardWrapper
