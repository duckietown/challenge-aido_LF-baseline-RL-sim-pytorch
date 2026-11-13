"""Algorithm helpers for RL training and evaluation."""

import numpy as np
from gymnasium import Env

from .learner import LearnerPolicy


def evaluate_policy(
    environment: Env,
    policy: LearnerPolicy,
    evaluation_episodes: int = 10,
) -> float:
    """Evaluate the policy in the given environment."""
    if evaluation_episodes <= 0:
        return 0
    average_reward = 0
    for _ in range(evaluation_episodes):
        observation, _ = environment.reset()
        episode_over = False
        while not episode_over:
            observation_array = np.asarray(observation)
            action = policy.predict(observation_array)
            observation, reward, terminated, truncated, _ = environment.step(
                action,
            )
            episode_over = terminated or truncated
            average_reward += float(reward)
    return average_reward / evaluation_episodes
