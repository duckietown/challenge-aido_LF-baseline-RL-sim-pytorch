"""Test the reinforcement learning agent."""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import cast

import numpy as np
from gymnasium import Env, spaces
from gymnasium.wrappers import TimeLimit

from solution.policy import DDPG
from training.utils import launch_environment
from training.utils.wrappers import (
    ActionWrapper,
    ImageWrapper,
    NormalizeWrapper,
    ResizeWrapper,
    RewardWrapper,
)

logging.basicConfig()
logger = logging.getLogger(__name__)


def _get_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--environment-id",
        type=str,
        default=None,
        help="Optional Gymnasium environment id to override the default.",
    )
    parser.add_argument(
        "--network-type",
        choices=("cnn", "dense"),
        default="cnn",
        help="Policy network architecture.",
    )
    parser.add_argument(
        "--max-episode-steps",
        default=100,
        type=int,
        help="Max episode steps",
    )
    parser.add_argument(
        "--episodes",
        default=0,
        type=int,
        help="Number of episodes to run. Use 0 to run until interrupted.",
    )
    parser.add_argument(
        "--max-total-steps",
        default=0,
        type=int,
        help="Maximum total steps to execute. Use 0 for no limit.",
    )
    parser.add_argument(
        "--model-directory",
        type=str,
        default="models",
        help="Model directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ddpg",
        help="Base filename used when loading model checkpoints.",
    )
    return parser.parse_args()


def _get_environment(
    max_episode_steps: int,
    environment_id: str | None,
) -> Env:
    environment = launch_environment(environment_id)
    environment = ResizeWrapper(environment)
    environment = NormalizeWrapper(environment)
    environment = ImageWrapper(environment)
    environment = ActionWrapper(environment)
    environment = RewardWrapper(environment)
    return TimeLimit(environment, max_episode_steps)


def _get_directory_path(directory: str) -> Path:
    normalized_directory = directory.rstrip("/")
    if not normalized_directory:
        normalized_directory = "."
    return Path(normalized_directory)


def main() -> None:
    """Run the saved RL policy for evaluation episodes."""
    arguments = _get_arguments()
    environment = _get_environment(
        arguments.max_episode_steps,
        arguments.environment_id,
    )
    try:
        observation_space = cast("spaces.Box", environment.observation_space)
        action_space = cast("spaces.Box", environment.action_space)
        state_dimensions = observation_space.shape
        action_dimensions = action_space.shape[0]
        max_action = np.asarray(action_space.high, dtype=np.float32)
        policy = DDPG(
            state_dimensions,
            action_dimensions,
            max_action,
            arguments.network_type,
        )
        model_directory = _get_directory_path(arguments.model_directory)
        model_directory_str = str(model_directory)
        policy.load(arguments.model_name, model_directory_str)
        observation, _ = environment.reset()
        episode_over = False
        total_steps = 0
        completed_episodes = 0
        run_forever = arguments.episodes == 0
        while run_forever or completed_episodes < arguments.episodes:
            while not episode_over:
                observation_array = np.asarray(observation, dtype=np.float32)
                action = policy.predict(observation_array)
                observation, _, terminated, truncated, _ = environment.step(
                    action,
                )
                episode_over = terminated or truncated
                total_steps += 1
                if (
                    arguments.max_total_steps > 0
                    and total_steps >= arguments.max_total_steps
                ):
                    logger.info("Stopping after %s total steps.", total_steps)
                    return
            completed_episodes += 1
            observation, _ = environment.reset()
            episode_over = False
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
