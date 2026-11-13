"""Train the reinforcement learning agent."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from gymnasium.wrappers import TimeLimit

from solution.policy import DDPG
from training.utils import (
    ReplayBuffer,
    evaluate_policy,
    launch_environment,
    seed,
)
from training.utils.wrappers import (
    ActionWrapper,
    ImageWrapper,
    NormalizeWrapper,
    ResizeWrapper,
    RewardWrapper,
)

if TYPE_CHECKING:
    from gymnasium import Env, spaces

logging.basicConfig()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ActionSelectionContext:
    action_dimensions: int
    action_space: spaces.Box
    arguments: Namespace
    generator: np.random.Generator
    max_action: np.ndarray
    min_action: np.ndarray
    policy: DDPG


@dataclass(frozen=True)
class _Transition:
    action: np.ndarray
    next_observation: np.ndarray
    observation: np.ndarray
    reward: float


@dataclass(frozen=True)
class _EvaluationContext:
    arguments: Namespace
    environment: Env
    model_directory: Path
    policy: DDPG
    results_directory: Path


@dataclass
class _EpisodeState:
    episode_number: int = 0
    episode_over: bool = True
    episode_reward: float = 0
    episode_timesteps: int = 0
    observation: np.ndarray | None = None
    timesteps_since_evaluation: int = 0
    total_timesteps: int = 0


@dataclass(frozen=True)
class _TrainingSession:
    action_context: _ActionSelectionContext
    environment: Env
    evaluation_context: _EvaluationContext
    evaluations: list[float]
    replay_buffer: ReplayBuffer


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
        "--seed",
        default=0,
        type=int,
        help="Sets Gym, PyTorch and NumPy seeds",
    )
    parser.add_argument(
        "--start-timesteps",
        default=10_000,
        type=int,
        help="How many timesteps the purely random policy is run for",
    )
    parser.add_argument(
        "--evaluation-frequency",
        default=5_000,
        type=int,
        help="How often (timesteps) we evaluate",
    )
    parser.add_argument(
        "--evaluation-episodes",
        default=10,
        type=int,
        help="Number of episodes to use during evaluation.",
    )
    parser.add_argument(
        "--max-timesteps",
        default=1_000_000,
        type=int,
        help="Max timesteps to run environment for",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Whether or not models are saved",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_false",
        dest="save_models",
        help="Disable model checkpoint saving.",
    )
    parser.add_argument(
        "--exploration-noise",
        default=0.1,
        type=float,
        help="Std of Gaussian exploration noise",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Batch size for both Actor and Critic",
    )
    parser.add_argument(
        "--discount",
        default=0.99,
        type=float,
        help="Discount factor",
    )
    parser.add_argument(
        "--tau",
        default=0.005,
        type=float,
        help="Target network update rate",
    )
    parser.add_argument(
        "--max-episode-steps",
        default=100,
        type=int,
        help="Max episode steps",
    )
    parser.add_argument(
        "--replay-buffer-max-size",
        default=10_000,
        type=int,
        help="Maximum number of steps to keep in the replay buffer",
    )
    parser.add_argument(
        "--results-directory",
        type=str,
        default="results",
        help="Results directory",
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
        help="Base filename used when saving model checkpoints.",
    )
    parser.set_defaults(save_models=True)
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


def _save_evaluations(
    results_directory: Path,
    evaluations: list[float],
) -> None:
    evaluation_array = np.asarray(evaluations, dtype=np.float32)
    np.savez(
        results_directory / "rewards.npz",
        evaluation_array,
    )


def _maybe_evaluate(
    context: _EvaluationContext,
    evaluations: list[float],
    total_timesteps: int,
    timesteps_since_evaluation: int,
) -> int:
    arguments = context.arguments
    if arguments.evaluation_episodes <= 0:
        return timesteps_since_evaluation
    if timesteps_since_evaluation < arguments.evaluation_frequency:
        return timesteps_since_evaluation
    timesteps_since_evaluation %= arguments.evaluation_frequency
    evaluation = evaluate_policy(
        context.environment,
        context.policy,
        arguments.evaluation_episodes,
    )
    evaluations.append(evaluation)
    logger.info(
        "Evaluation reward at step %s: %.3f",
        total_timesteps,
        evaluation,
    )
    if arguments.save_models:
        model_directory = str(context.model_directory)
        context.policy.save(arguments.model_name, model_directory)
    _save_evaluations(context.results_directory, evaluations)
    return timesteps_since_evaluation


def _finish_episode(
    context: _EvaluationContext,
    replay_buffer: ReplayBuffer,
    evaluations: list[float],
    state: _EpisodeState,
) -> int:
    observation = state.observation
    if observation is None:
        return state.timesteps_since_evaluation
    arguments = context.arguments
    logger.info(
        "Episode %s finished after %s steps with reward %.3f.",
        state.episode_number,
        state.episode_timesteps,
        state.episode_reward,
    )
    context.policy.train(
        replay_buffer,
        state.episode_timesteps,
        arguments.batch_size,
        arguments.discount,
        arguments.tau,
    )
    return _maybe_evaluate(
        context,
        evaluations,
        state.total_timesteps,
        state.timesteps_since_evaluation,
    )


def _select_action(
    context: _ActionSelectionContext,
    total_timesteps: int,
    observation: np.ndarray,
) -> np.ndarray:
    arguments = context.arguments
    if total_timesteps < arguments.start_timesteps:
        sampled_action = context.action_space.sample()
        return np.asarray(sampled_action, dtype=np.float32)
    action = context.policy.predict(observation)
    if arguments.exploration_noise != 0:
        action += context.generator.normal(
            0,
            arguments.exploration_noise,
            context.action_dimensions,
        )
    return np.clip(action, context.min_action, context.max_action)


def _initialize_training_session(arguments: Namespace) -> _TrainingSession:
    results_directory = _get_directory_path(arguments.results_directory)
    results_directory.mkdir(parents=True, exist_ok=True)
    model_directory = _get_directory_path(arguments.model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)
    generator = seed(arguments.seed)
    environment = _get_environment(
        arguments.max_episode_steps,
        arguments.environment_id,
    )
    observation_space = cast("spaces.Box", environment.observation_space)
    action_space = cast("spaces.Box", environment.action_space)
    action_space.seed(arguments.seed)
    state_dimensions = observation_space.shape
    action_dimensions = action_space.shape[0]
    min_action = np.asarray(action_space.low, dtype=np.float32)
    max_action = np.asarray(action_space.high, dtype=np.float32)
    policy = DDPG(
        state_dimensions,
        action_dimensions,
        max_action,
        arguments.network_type,
    )
    replay_buffer = ReplayBuffer(arguments.replay_buffer_max_size)
    evaluations: list[float] = []
    if arguments.evaluation_episodes > 0:
        evaluation = evaluate_policy(
            environment,
            policy,
            arguments.evaluation_episodes,
        )
        evaluations.append(evaluation)
    action_context = _ActionSelectionContext(
        action_dimensions=action_dimensions,
        action_space=action_space,
        arguments=arguments,
        generator=generator,
        max_action=max_action,
        min_action=min_action,
        policy=policy,
    )
    evaluation_context = _EvaluationContext(
        arguments=arguments,
        environment=environment,
        model_directory=model_directory,
        policy=policy,
        results_directory=results_directory,
    )
    return _TrainingSession(
        action_context=action_context,
        environment=environment,
        evaluation_context=evaluation_context,
        evaluations=evaluations,
        replay_buffer=replay_buffer,
    )


def _reset_episode(
    arguments: Namespace,
    environment: Env,
    state: _EpisodeState,
) -> None:
    reset_seed = arguments.seed + state.episode_number
    observation, _ = environment.reset(seed=reset_seed)
    state.observation = np.asarray(observation, dtype=np.float32)
    state.episode_number += 1
    state.episode_over = False
    state.episode_reward = 0
    state.episode_timesteps = 0


def _require_observation(observation: np.ndarray | None) -> np.ndarray:
    if observation is not None:
        return observation
    message = "Expected an observation after resetting the environment."
    raise RuntimeError(message)


def _record_transition(
    replay_buffer: ReplayBuffer,
    transition: _Transition,
    *,
    episode_over: bool,
) -> None:
    current_observation = np.asarray(transition.observation, dtype=np.float32)
    current_next_observation = np.asarray(
        transition.next_observation,
        dtype=np.float32,
    )
    current_action = np.asarray(transition.action, dtype=np.float32)
    replay_buffer.add(
        current_observation,
        current_next_observation,
        current_action,
        float(transition.reward),
        episode_over=episode_over,
    )


def _finish_training(
    context: _EvaluationContext,
    evaluations: list[float],
) -> None:
    if context.arguments.save_models:
        model_directory = str(context.model_directory)
        context.policy.save(context.arguments.model_name, model_directory)
    if evaluations:
        _save_evaluations(context.results_directory, evaluations)


def _run_training_loop(
    arguments: Namespace,
    session: _TrainingSession,
) -> None:
    state = _EpisodeState()
    logger.info("Starting DDPG training loop.")
    while state.total_timesteps < arguments.max_timesteps:
        if state.episode_over:
            state.timesteps_since_evaluation = _finish_episode(
                session.evaluation_context,
                session.replay_buffer,
                session.evaluations,
                state,
            )
            _reset_episode(arguments, session.environment, state)

        observation = _require_observation(state.observation)
        action = _select_action(
            session.action_context,
            state.total_timesteps,
            observation,
        )
        next_observation, reward, terminated, truncated, _ = (
            session.environment.step(action)
        )
        state.episode_over = terminated or truncated
        state.episode_reward += float(reward)
        _record_transition(
            session.replay_buffer,
            _Transition(
                action=action,
                next_observation=next_observation,
                observation=observation,
                reward=float(reward),
            ),
            episode_over=state.episode_over,
        )
        state.observation = np.asarray(next_observation, dtype=np.float32)
        state.episode_timesteps += 1
        state.total_timesteps += 1
        state.timesteps_since_evaluation += 1
    _finish_training(session.evaluation_context, session.evaluations)


def main() -> None:
    """Train the RL baseline with the configured hyperparameters."""
    arguments = _get_arguments()
    session = _initialize_training_session(arguments)
    try:
        _run_training_loop(arguments, session)
    finally:
        session.environment.close()


if __name__ == "__main__":
    main()
