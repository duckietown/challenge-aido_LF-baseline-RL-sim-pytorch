"""Dataset-style storage helpers for RL training."""

import secrets

import numpy as np
from numpy.random import Generator


class ReplayBuffer:
    """A simple replay buffer for storing transitions."""

    _generator: Generator
    _max_size: int
    _storage: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]]

    def __init__(self, max_size: int) -> None:
        """Initialize the replay buffer storage and sampler."""
        self._storage = []
        self._max_size = max_size
        self._generator = np.random.default_rng()

    def add(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        *,
        episode_over: bool,
    ) -> None:
        """Store one transition in the replay buffer."""
        storage_length = len(self._storage)
        if storage_length < self._max_size:
            self._storage.append(
                (state, next_state, action, reward, episode_over),
            )
            return
        random_integer = secrets.randbelow(storage_length)
        self._storage.pop(random_integer)
        self._storage.append(
            (state, next_state, action, reward, episode_over),
        )

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return len(self._storage)

    def sample(
        self,
        batch_size: int = 100,
        *,
        flat: bool = True,
    ) -> dict[str, np.ndarray]:
        """Sample a batch of transitions from the replay buffer."""
        storage_length = len(self._storage)
        random_integers = self._generator.integers(
            0,
            storage_length,
            batch_size,
        )
        states: list[np.ndarray] = []
        next_states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        dones: list[np.ndarray] = []
        for random_integer in random_integers:
            state, next_state, action, reward, done = self._storage[
                random_integer
            ]
            state_array = np.asarray(state)
            next_state_array = np.asarray(next_state)
            if flat:
                state_array = state_array.flatten()
                next_state_array = next_state_array.flatten()
            states.append(state_array)
            next_states.append(next_state_array)
            actions.append(np.asarray(action))
            rewards.append(np.asarray(reward))
            dones.append(np.asarray(done))
        state_array = np.stack(states)
        next_state_array = np.stack(next_states)
        action_array = np.stack(actions)
        reward_array = np.stack(rewards).reshape(-1, 1)
        done_array = np.stack(dones).reshape(-1, 1)
        return {
            "state": state_array,
            "next_state": next_state_array,
            "action": action_array,
            "reward": reward_array,
            "done": done_array,
        }
