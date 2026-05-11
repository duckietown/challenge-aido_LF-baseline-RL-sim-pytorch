"""Run the RL baseline policy inside Duckiematrix."""

import io
import logging
import os
from threading import Event
from typing import Any

import numpy as np
import torch
from gym_duckiematrix.gym_environment import GymEnvironment
from PIL import Image

from solution.action_wrapper import DTPytorchActionWrapper
from solution.policy import DDPG
from solution.wrappers import DTPytorchWrapper

ENGINE_HOST = os.environ.get("DUCKIEMATRIX_ENGINE_HOSTNAME", "127.0.0.1")
ENGINE_PORT_STRING = os.environ.get("DUCKIEMATRIX_ENGINE_PORT")
if ENGINE_PORT_STRING is None:
    ENGINE_PORT_STRING = os.environ.get("ENGINE_PORT", "7501")
ENGINE_PORT = int(ENGINE_PORT_STRING)
EXPECTED_IMAGE_DIMENSIONS = 3
EXPECTED_IMAGE_DTYPE = np.uint8

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _clip_pwm(value: float) -> float:
    clipped_value = np.clip(value, -1, +1)
    return float(clipped_value)


class PytorchBaselineAgent:
    """Bridge Duckiematrix observations into the RL baseline policy."""

    _action_processor: DTPytorchActionWrapper
    _environment: GymEnvironment
    _load_model: bool
    _model_path: str | None
    _policy: DDPG
    _preprocessor: DTPytorchWrapper
    _shutdown_event: Event
    _step_count: int
    _vehicle_name: str

    def __init__(
        self,
        *,
        load_model: bool = True,
        model_path: str | None = None,
    ) -> None:
        """Initialize the RL policy, model state, and environment."""
        self._load_model = load_model
        self._model_path = model_path
        self._shutdown_event = Event()
        self._step_count = 0

        self._preprocessor = DTPytorchWrapper()
        self._action_processor = DTPytorchActionWrapper()
        max_action = np.asarray([1, 1], dtype=np.float32)
        self._policy = DDPG(
            state_dimensions=self._preprocessor.shape,
            action_dimensions=2,
            max_action=max_action,
            network_type="cnn",
        )

        self._check_gpu_available()
        if self._load_model:
            model_name = self._model_path or "model"
            logger.info("Loading models from %s", model_name)
            self._policy.load(model_name, "models", for_inference=True)

        self._environment = GymEnvironment(host=ENGINE_HOST, port=ENGINE_PORT)
        self._vehicle_name = self._resolve_vehicle_name()

    def _check_gpu_available(self) -> None:
        available = torch.cuda.is_available()
        logger.info("torch.cuda.is_available = %r", available)
        if available:
            device_index = torch.cuda.current_device()
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(device_index)
            logger.info(
                "device %s of %s; name = %r",
                device_index,
                device_count,
                device_name,
            )

    def _resolve_vehicle_name(self) -> str:
        preferred_name = os.environ.get("VEHICLE_NAME", "")
        available_vehicle_names = self._environment.vehicle_names
        if preferred_name and preferred_name in available_vehicle_names:
            return preferred_name
        if preferred_name:
            logger.warning(
                "VEHICLE_NAME=%s was not discovered. Falling back to %s.",
                preferred_name,
                available_vehicle_names[0],
            )
        return available_vehicle_names[0]

    def _predict_action(self, processed_observation: np.ndarray) -> np.ndarray:
        predicted_action = self._policy.predict(processed_observation)
        return np.asarray(predicted_action, dtype=np.float32)

    def _compute_pwm(self, image_data: bytes) -> tuple[float, float]:
        observation = _jpg_to_rgb(image_data)
        processed_observation = self._preprocessor.preprocess(observation)
        action = self._predict_action(processed_observation)
        pwm_left, pwm_right = self._action_processor.convert(action)
        clipped_left = _clip_pwm(pwm_left)
        clipped_right = _clip_pwm(pwm_right)
        return clipped_left, clipped_right

    def _callback(self, world_input: dict[str, Any]) -> None:
        entities = world_input.get("entities", {})
        entity = entities.get(self._vehicle_name, {})
        compressed_image = entity.get("compressed_image") or {}
        raw_image_data = compressed_image.get("data")
        image_data = _normalize_image_bytes(raw_image_data)

        if image_data is None:
            pwm_left, pwm_right = 0, 0
        else:
            pwm_left, pwm_right = self._compute_pwm(image_data)

        self._step_count += 1
        if self._step_count % 50 == 0:
            logger.info(
                "step %d pwm=(%.3f, %.3f)",
                self._step_count,
                pwm_left,
                pwm_right,
            )

        wheel_pwm = (pwm_left, pwm_right)
        action_map = {self._vehicle_name: wheel_pwm}
        self._environment.step(action_map)

    def run(self) -> None:
        """Start the environment bridge and wait for shutdown."""
        logger.info(
            "RL baseline agent: engine=%s:%d",
            ENGINE_HOST,
            ENGINE_PORT,
        )
        available_vehicle_names = self._environment.vehicle_names
        logger.info(
            "RL baseline agent: vehicles=%s",
            available_vehicle_names,
        )
        self._environment.attach(self._callback)
        self._environment.start()
        try:
            self._shutdown_event.wait()
        finally:
            self._environment.stop()


def _normalize_image_bytes(image_data: object) -> bytes | None:
    normalized: bytes | None = None
    if isinstance(image_data, bytes):
        normalized = image_data
    elif isinstance(image_data, bytearray):
        normalized = bytes(image_data)
    elif isinstance(image_data, memoryview):
        normalized = image_data.tobytes()
    elif isinstance(image_data, list) and image_data:
        try:
            normalized = bytes(image_data)
        except ValueError:
            normalized = None
    return normalized or None


def _jpg_to_rgb(image_data: bytes) -> np.ndarray:
    image_buffer = io.BytesIO(image_data)
    with Image.open(image_buffer) as image:
        rgb_image = image.convert("RGB")
        data = np.array(rgb_image)
    if data.ndim != EXPECTED_IMAGE_DIMENSIONS:
        message = f"Expected a 3D image array, got shape {data.shape}."
        raise ValueError(message)
    if data.dtype != EXPECTED_IMAGE_DTYPE:
        message = (
            f"Expected image dtype {EXPECTED_IMAGE_DTYPE}, "
            f"got {data.dtype}."
        )
        raise TypeError(message)
    return data


def main() -> None:
    """Instantiate and run the RL baseline agent."""
    agent = PytorchBaselineAgent(load_model=True, model_path=None)
    agent.run()


if __name__ == "__main__":
    main()
