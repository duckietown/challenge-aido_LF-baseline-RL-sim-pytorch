"""Deep Deterministic Policy Gradients (DDPG).

Paper: https://arxiv.org/abs/1509.02971
"""

from __future__ import annotations

import functools
import logging
import operator
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, cuda, device
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ModuleList,
    functional,
)
from torch.optim import Adam

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from training.utils.dataset import ReplayBuffer

IMAGE_CHANNELS = 3
LEARNING_RATE = 0.0001

CNN_CONFIGURATIONS = {
    (120, 160): (((8, 2), (4, 2), (4, 2), (4, 1)), 32 * 9 * 14),
    (64, 64): (((4, 2), (2, 2), (2, 2), (2, 1)), 32 * 6 * 6),
}

device_type = "cuda" if cuda.is_available() else "cpu"
device_ = device(device_type)


def _flatten_state_dimensions(state_dimensions: int | Sequence[int]) -> int:
    if isinstance(state_dimensions, int):
        return state_dimensions
    return functools.reduce(operator.mul, state_dimensions, 1)


def _normalize_image_state_dimensions(
    state_dimensions: Sequence[int],
) -> tuple[int, int, int]:
    if len(state_dimensions) != IMAGE_CHANNELS:
        message = (
            "Expected image state dimensions to have length 3, got "
            f"{tuple(state_dimensions)}."
        )
        raise ValueError(message)
    dimensions = tuple(int(dimension) for dimension in state_dimensions)
    if dimensions[0] == IMAGE_CHANNELS:
        return dimensions[0], dimensions[1], dimensions[2]
    if dimensions[2] == IMAGE_CHANNELS:
        height, width, channels = dimensions
        return channels, height, width
    message = (
        "Expected image state dimensions to use CHW or HWC layout with "
        f"3 channels, got {dimensions}."
    )
    raise ValueError(message)


def _get_cnn_configuration(
    state_dimensions: Sequence[int],
) -> tuple[tuple[tuple[int, int], ...], int]:
    _, height, width = _normalize_image_state_dimensions(state_dimensions)
    resolution = (height, width)
    configuration = CNN_CONFIGURATIONS.get(resolution)
    if configuration is None:
        supported_resolutions = ", ".join(
            f"{height}x{width}" for height, width in CNN_CONFIGURATIONS
        )
        message = (
            "Expected image state dimensions to resolve to one of "
            f"{supported_resolutions}, got {resolution[0]}x{resolution[1]}."
        )
        raise ValueError(message)
    return configuration


def _build_convolutions(
    state_dimensions: Sequence[int],
) -> tuple[ModuleList, int]:
    convolution_parameters, flat_size = _get_cnn_configuration(
        state_dimensions,
    )
    convolutions = ModuleList(
        Conv2d(IMAGE_CHANNELS, 32, kernel_size, stride=stride)
        if index == 0
        else Conv2d(32, 32, kernel_size, stride=stride)
        for index, (kernel_size, stride) in enumerate(convolution_parameters)
    )
    return convolutions, flat_size


def _normalize_image_state_array(state_array: np.ndarray) -> np.ndarray:
    if state_array.ndim != IMAGE_CHANNELS:
        message = (
            "Expected image state to have 3 dimensions, got "
            f"{state_array.shape}."
        )
        raise ValueError(message)
    if state_array.shape[0] == IMAGE_CHANNELS:
        return state_array
    if state_array.shape[2] == IMAGE_CHANNELS:
        return state_array.transpose(2, 0, 1)
    message = (
        "Expected image state to have shape (3, H, W) or (H, W, 3), got "
        f"{state_array.shape}."
    )
    raise ValueError(message)


def _require_image_state_dimensions(
    state_dimensions: int | Sequence[int],
) -> tuple[int, int, int]:
    if isinstance(state_dimensions, int):
        message = (
            "Expected CNN state dimensions to be a 3D image shape, got "
            f"{state_dimensions}."
        )
        raise TypeError(message)
    return _normalize_image_state_dimensions(state_dimensions)


def _load_module_state_dictionary(
    module: Module,
    state_dictionary: dict[str, Tensor],
    *,
    allow_missing_max_action: bool = False,
) -> None:
    incompatible_keys = module.load_state_dict(
        state_dictionary,
        strict=False,
    )
    missing_keys = set(incompatible_keys.missing_keys)
    if allow_missing_max_action:
        missing_keys.discard("_max_action")
    unexpected_keys = set(incompatible_keys.unexpected_keys)
    if missing_keys or unexpected_keys:
        details = []
        if missing_keys:
            details.append(f"missing keys: {sorted(missing_keys)}")
        if unexpected_keys:
            details.append(f"unexpected keys: {sorted(unexpected_keys)}")
        message = "Could not load DDPG state dictionary; " + "; ".join(details)
        raise RuntimeError(message)


def _normalize_max_action(
    max_action: np.ndarray,
    action_dimensions: int,
) -> np.ndarray:
    action_limits = np.asarray(max_action, dtype=np.float32)
    if action_limits.ndim == 0:
        action_limit_value = float(action_limits)
        return np.full(
            action_dimensions,
            action_limit_value,
            dtype=np.float32,
        )
    if action_limits.shape != (action_dimensions,):
        message = (
            "Expected max_action to have shape "
            f"({action_dimensions},), got {action_limits.shape}."
        )
        raise ValueError(message)
    return action_limits


def _scale_actor_output(logits: Tensor, max_action: Tensor) -> Tensor:
    throttle = max_action[:1] * torch.sigmoid(logits[:, :1])
    if logits.shape[1] == 1:
        return throttle
    steering = max_action[1:] * torch.tanh(logits[:, 1:])
    return torch.cat((throttle, steering), dim=1)


class ActorCNN(Module):
    """Actor network for continuous action spaces."""

    _affine_linear_transformations: ModuleList
    _batch_normalizations: ModuleList
    _convolutions: ModuleList
    _dropout: Dropout
    _leaky_rectified_linear_unit: LeakyReLU
    _max_action: Tensor

    def __init__(
        self,
        state_dimensions: Sequence[int],
        action_dimensions: int,
        max_action: np.ndarray,
    ) -> None:
        """Initialize the convolutional actor network."""
        super().__init__()
        self._leaky_rectified_linear_unit = LeakyReLU()
        convolutions, flat_size = _build_convolutions(state_dimensions)
        self._convolutions = convolutions
        batch_normalization_1 = BatchNorm2d(32)
        batch_normalization_2 = BatchNorm2d(32)
        batch_normalization_3 = BatchNorm2d(32)
        batch_normalization_4 = BatchNorm2d(32)
        self._batch_normalizations = ModuleList(
            (
                batch_normalization_1,
                batch_normalization_2,
                batch_normalization_3,
                batch_normalization_4,
            ),
        )
        self._dropout = Dropout(0.5)
        affine_linear_transformation_1 = Linear(flat_size, 512)
        affine_linear_transformation_2 = Linear(512, action_dimensions)
        self._affine_linear_transformations = ModuleList(
            (
                affine_linear_transformation_1,
                affine_linear_transformation_2,
            ),
        )
        max_action_array = _normalize_max_action(max_action, action_dimensions)
        max_action_tensor = torch.as_tensor(
            max_action_array,
            dtype=torch.float32,
        )
        self.register_buffer(
            "_max_action",
            max_action_tensor,
        )

    def forward(self, state: Tensor) -> Tensor:
        """Predict one bounded action batch from image observations."""
        x = state
        for i in range(4):
            x = self._convolutions[i](x)
            x = self._leaky_rectified_linear_unit(x)
            x = self._batch_normalizations[i](x)
        size = x.size(0)
        x = x.reshape(size, -1)
        x = self._dropout(x)
        x = self._affine_linear_transformations[0](x)
        x = self._leaky_rectified_linear_unit(x)
        x = self._affine_linear_transformations[1](x)
        return _scale_actor_output(x, self._max_action)


class ActorDense(Module):
    """Actor network for continuous action spaces."""

    _affine_linear_transformations: ModuleList
    _max_action: Tensor

    def __init__(
        self,
        state_dimensions: int | Sequence[int],
        action_dimensions: int,
        max_action: np.ndarray,
    ) -> None:
        """Initialize the dense actor network."""
        super().__init__()
        flat_state_dimensions = _flatten_state_dimensions(state_dimensions)
        affine_linear_transformation_1 = Linear(flat_state_dimensions, 400)
        affine_linear_transformation_2 = Linear(400, 300)
        affine_linear_transformation_3 = Linear(300, action_dimensions)
        self._affine_linear_transformations = ModuleList(
            (
                affine_linear_transformation_1,
                affine_linear_transformation_2,
                affine_linear_transformation_3,
            ),
        )
        max_action_array = _normalize_max_action(max_action, action_dimensions)
        max_action_tensor = torch.as_tensor(
            max_action_array,
            dtype=torch.float32,
        )
        self.register_buffer(
            "_max_action",
            max_action_tensor,
        )

    def forward(self, state: Tensor) -> Tensor:
        """Predict one bounded action batch from flat observations."""
        x = self._affine_linear_transformations[0](state)
        x = functional.relu(x)
        x = self._affine_linear_transformations[1](x)
        x = functional.relu(x)
        x = self._affine_linear_transformations[2](x)
        return _scale_actor_output(x, self._max_action)


class CriticCNN(Module):
    """Critic network for continuous action spaces."""

    _affine_linear_transformations: ModuleList
    _batch_normalizations: ModuleList
    _convolutions: ModuleList
    _dropout: Dropout
    _leaky_rectified_linear_unit: LeakyReLU

    def __init__(
        self,
        state_dimensions: Sequence[int],
        action_dimensions: int,
    ) -> None:
        """Initialize the convolutional critic network."""
        super().__init__()
        self._leaky_rectified_linear_unit = LeakyReLU()
        convolutions, flat_size = _build_convolutions(state_dimensions)
        self._convolutions = convolutions
        batch_normalization_1 = BatchNorm2d(32)
        batch_normalization_2 = BatchNorm2d(32)
        batch_normalization_3 = BatchNorm2d(32)
        batch_normalization_4 = BatchNorm2d(32)
        self._batch_normalizations = ModuleList(
            (
                batch_normalization_1,
                batch_normalization_2,
                batch_normalization_3,
                batch_normalization_4,
            ),
        )
        self._dropout = Dropout(0.5)
        affine_linear_transformation_1 = Linear(flat_size, 256)
        affine_linear_transformation_2 = Linear(256 + action_dimensions, 128)
        affine_linear_transformation_3 = Linear(128, 1)
        self._affine_linear_transformations = ModuleList(
            (
                affine_linear_transformation_1,
                affine_linear_transformation_2,
                affine_linear_transformation_3,
            ),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """Estimate one Q-value batch from image observations."""
        x = state
        for i in range(4):
            x = self._convolutions[i](x)
            x = self._leaky_rectified_linear_unit(x)
            x = self._batch_normalizations[i](x)
        size = x.size(0)
        x = x.reshape(size, -1)
        x = self._affine_linear_transformations[0](x)
        x = self._leaky_rectified_linear_unit(x)
        x = torch.cat((x, action), 1)
        x = self._affine_linear_transformations[1](x)
        x = self._leaky_rectified_linear_unit(x)
        return self._affine_linear_transformations[2](x)


class CriticDense(Module):
    """Critic network for continuous action spaces."""

    _affine_linear_transformations: ModuleList

    def __init__(
        self,
        state_dimensions: int | Sequence[int],
        action_dimensions: int,
    ) -> None:
        """Initialize the dense critic network."""
        super().__init__()
        flat_state_dimensions = _flatten_state_dimensions(state_dimensions)
        affine_linear_transformation_1 = Linear(flat_state_dimensions, 400)
        affine_linear_transformation_2 = Linear(400 + action_dimensions, 300)
        affine_linear_transformation_3 = Linear(300, 1)
        self._affine_linear_transformations = ModuleList(
            (
                affine_linear_transformation_1,
                affine_linear_transformation_2,
                affine_linear_transformation_3,
            ),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """Estimate one Q-value batch from flat observations."""
        x = self._affine_linear_transformations[0](state)
        x = functional.relu(x)
        x = torch.cat((x, action), 1)
        x = self._affine_linear_transformations[1](x)
        x = functional.relu(x)
        return self._affine_linear_transformations[2](x)


class DDPG:
    """Deep Deterministic Policy Gradient (DDPG) agent."""

    _actor: ActorCNN | ActorDense
    _actor_optimizer: Adam
    _actor_target: ActorCNN | ActorDense
    _critic: CriticCNN | CriticDense
    _critic_optimizer: Adam
    _critic_target: CriticCNN | CriticDense
    _flat: bool

    def __init__(
        self,
        state_dimensions: int | Sequence[int],
        action_dimensions: int,
        max_action: np.ndarray,
        network_type: str,
    ) -> None:
        """Initialize the DDPG actor, critic, and target networks."""
        super().__init__()
        if network_type not in ("cnn", "dense"):
            message = (
                "Expected network_type to be one of ('cnn', 'dense'), got "
                f"{network_type}."
            )
            raise ValueError(message)
        if network_type == "dense":
            self._flat = True
            actor_dense = ActorDense(
                state_dimensions,
                action_dimensions,
                max_action,
            )
            self._actor = actor_dense.to(device_)
            actor_dense = ActorDense(
                state_dimensions,
                action_dimensions,
                max_action,
            )
            self._actor_target = actor_dense.to(device_)
        else:
            self._flat = False
            image_state_dimensions = _require_image_state_dimensions(
                state_dimensions,
            )
            actor_cnn = ActorCNN(
                image_state_dimensions,
                action_dimensions,
                max_action,
            )
            self._actor = actor_cnn.to(device_)
            actor_cnn = ActorCNN(
                image_state_dimensions,
                action_dimensions,
                max_action,
            )
            self._actor_target = actor_cnn.to(device_)
        state_dictionary = self._actor.state_dict()
        self._actor_target.load_state_dict(state_dictionary)
        parameters = self._actor.parameters()
        self._actor_optimizer = Adam(parameters, LEARNING_RATE)
        if network_type == "dense":
            critic_dense = CriticDense(state_dimensions, action_dimensions)
            self._critic = critic_dense.to(device_)
            critic_dense = CriticDense(state_dimensions, action_dimensions)
            self._critic_target = critic_dense.to(device_)
        else:
            critic_cnn = CriticCNN(
                image_state_dimensions,
                action_dimensions,
            )
            self._critic = critic_cnn.to(device_)
            critic_cnn = CriticCNN(
                image_state_dimensions,
                action_dimensions,
            )
            self._critic_target = critic_cnn.to(device_)
        state_dictionary = self._critic.state_dict()
        self._critic_target.load_state_dict(state_dictionary)
        parameters = self._critic.parameters()
        self._critic_optimizer = Adam(parameters, LEARNING_RATE)

    def load(
        self,
        filename: str,
        directory: str,
        *,
        for_inference: bool = False,
    ) -> None:
        """Load actor and critic checkpoints from disk."""
        state_dictionary = torch.load(
            f"{directory}/{filename}_actor.pth",
            map_location=device_,
        )
        _load_module_state_dictionary(
            self._actor,
            state_dictionary,
            allow_missing_max_action=True,
        )
        _load_module_state_dictionary(
            self._actor_target,
            state_dictionary,
            allow_missing_max_action=True,
        )
        state_dictionary = torch.load(
            f"{directory}/{filename}_critic.pth",
            map_location=device_,
        )
        _load_module_state_dictionary(self._critic, state_dictionary)
        _load_module_state_dictionary(self._critic_target, state_dictionary)
        if for_inference:
            self._actor.eval()
            self._actor_target.eval()
            self._critic.eval()
            self._critic_target.eval()

    def predict(self, state_array: np.ndarray) -> np.ndarray:
        """Predict one action from the provided state array."""
        state_array = np.asarray(state_array, dtype=np.float32)
        if self._flat:
            state_array = state_array.reshape(1, -1)
        else:
            state_array = _normalize_image_state_array(state_array)
            state_array = np.expand_dims(state_array, 0)
        state = torch.as_tensor(
            state_array,
            dtype=torch.float32,
            device=device_,
        )
        with torch.no_grad():
            action = self._actor(state)
        action = action.cpu()
        action_array = action.numpy()
        return action_array.flatten()

    def save(self, filename: str, directory: str) -> None:
        """Persist the actor and critic checkpoints to disk."""
        state_dictionary = self._actor.state_dict()
        torch.save(state_dictionary, f"{directory}/{filename}_actor.pth")
        state_dictionary = self._critic.state_dict()
        torch.save(state_dictionary, f"{directory}/{filename}_critic.pth")

    def train(
        self,
        replay_buffer: ReplayBuffer,
        iterations: int,
        batch_size: int = 64,
        discount: float = 0.99,
        tau: float = 0.001,
    ) -> None:
        """Run the requested number of DDPG optimization steps."""
        if len(replay_buffer) < batch_size:
            logger.debug(
                "Skipping DDPG update because the replay buffer has only %s "
                "transitions.",
                len(replay_buffer),
            )
            return
        for _ in range(iterations):
            sample = replay_buffer.sample(batch_size, flat=self._flat)
            state = torch.as_tensor(
                sample["state"],
                dtype=torch.float32,
                device=device_,
            )
            action = torch.as_tensor(
                sample["action"],
                dtype=torch.float32,
                device=device_,
            )
            next_state = torch.as_tensor(
                sample["next_state"],
                dtype=torch.float32,
                device=device_,
            )
            done = torch.as_tensor(
                1 - sample["done"],
                dtype=torch.float32,
                device=device_,
            )
            reward = torch.as_tensor(
                sample["reward"],
                dtype=torch.float32,
                device=device_,
            )
            with torch.no_grad():
                target_action = self._actor_target(next_state)
                target_q = self._critic_target(next_state, target_action)
                target_q = reward + done * discount * target_q
            current_q = self._critic(state, action)
            critic_loss = functional.mse_loss(current_q, target_q)
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            self._critic_optimizer.step()
            action = self._actor(state)
            q: Tensor = self._critic(state, action)
            actor_loss = -q.mean()
            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            self._actor_optimizer.step()
            with torch.no_grad():
                for parameter, target_parameter in zip(
                    self._critic.parameters(),
                    self._critic_target.parameters(),
                    strict=False,
                ):
                    target_parameter.lerp_(parameter, tau)
                for parameter, target_parameter in zip(
                    self._actor.parameters(),
                    self._actor_target.parameters(),
                    strict=False,
                ):
                    target_parameter.lerp_(parameter, tau)
