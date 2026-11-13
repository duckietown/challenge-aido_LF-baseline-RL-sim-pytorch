"""Learner interfaces and seeding helpers for RL training."""

import random
from typing import Protocol

import numpy as np
import torch
from numpy.random import Generator


class LearnerPolicy(Protocol):
    """Policy interface required by RL evaluation helpers."""

    def predict(self, state_array: np.ndarray) -> np.ndarray:
        """Predict one action for the provided observation."""


def seed(seed_value: int) -> Generator:
    """Set the seed for random number generation."""
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    return np.random.default_rng(seed_value)
