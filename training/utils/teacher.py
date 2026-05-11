# pyright: reportMissingTypeStubs=false

"""Pure-pursuit expert policy used by imitation trainers."""

from typing import Protocol, cast

import numpy as np
from duckietown.sdk.utils.common import get_right_vector

POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.8
GAIN = 10
FOLLOWING_DISTANCE = 0.3


class _PurePursuitEnvironment(Protocol):
    """Environment interface required by the pure-pursuit expert."""

    cur_pos: np.ndarray
    cur_angle: float
    unwrapped: "_PurePursuitEnvironment"

    def closest_curve_point(
        self,
        position: np.ndarray,
        angle: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the nearest curve point and tangent."""


class PurePursuitPolicy:
    """Pure-pursuit controller used as a lightweight driving expert."""

    _env: _PurePursuitEnvironment
    _following_distance: float
    _max_iterations: int
    _position_threshold: float
    _ref_velocity: float

    def __init__(
        self,
        env: object,
        ref_velocity: float = REF_VELOCITY,
        position_threshold: float = POSITION_THRESHOLD,
        following_distance: float = FOLLOWING_DISTANCE,
        max_iterations: int = 1000,
    ) -> None:
        """Store the environment handle and pure-pursuit parameters."""
        typed_env = cast("_PurePursuitEnvironment", env)
        self._env = typed_env.unwrapped
        self._following_distance = following_distance
        self._max_iterations = max_iterations
        self._position_threshold = position_threshold
        self._ref_velocity = ref_velocity

    def predict(
        self,
        observation: object | None,
    ) -> tuple[float, float]:
        """Return the teacher action for the provided observation."""
        del observation
        closest_point, closest_tangent = self._env.closest_curve_point(
            self._env.cur_pos,
            self._env.cur_angle,
        )
        if closest_point is None or closest_tangent is None:
            return 0, 0

        iterations = 0
        lookup_distance = self._following_distance
        curve_point = None
        while iterations < self._max_iterations:
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = self._env.closest_curve_point(
                follow_point,
                self._env.cur_angle,
            )
            if curve_point is not None:
                break
            iterations += 1
            lookup_distance *= 0.5
        if curve_point is None:
            return 0, 0

        point_vec = curve_point - self._env.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        right_vector = get_right_vector(self._env.cur_angle)
        dot = np.dot(right_vector, point_vec)
        steering = GAIN * -dot

        return self._ref_velocity, steering
