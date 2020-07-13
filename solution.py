#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from aido_schemas import EpisodeStart, protocol_agent_duckiebot1, PWMCommands, Duckiebot1Commands, LEDSCommands, RGB, \
    wrap_direct, Context, Duckiebot1Observations, JPGImage

from model import DDPG
from wrappers import *

class FakeWrap:
    def __init__(self):
        self.env = None
        self.action_space = None

        self.camera_width = 640
        self.camera_height = 480
        self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.camera_height, self.camera_width, 3),
                dtype=np.uint8
        )
        self.reward_range = None
        self.metadata = None


class DTPytorchWrapper():
    def __init__(self, shape=(120, 160, 3)):
        self.shape = shape
        self.transposed_shape = (shape[2], shape[0], shape[1])

    def preprocess(self, obs):
        # from PIL import Image
        # return np.array(Image.fromarray(obs).resize(self.shape[0:2])).transpose(2, 0, 1)

        import cv2
        obs = cv2.resize(obs, self.shape[0:2])
        # NOTICE: OpenCV changes the order of the channels !!!
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        return obs.transpose(2, 0, 1)

class PytorchRLBaseline:
    def __init__(self, load_model=False, model_path=None):
        self.image_processor = DTPytorchWrapper()
        self.action_processor = ActionWrapper(FakeWrap())

        self.model = DDPG(state_dim=self.image_processor.shape, action_dim=2, max_action=1, net_type="cnn")
        self.current_image = np.zeros((640, 480, 3))

        # TODO: Uncomment when you train your model!

        # fp = model_path if model_path else "model"
        # self.model.load(fp, "models", for_inference=True)

    def init(self, context: Context):
        context.info('init()')

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, data: Duckiebot1Observations):
        camera: JPGImage = data.camera
        obs = jpg2rgb(camera.jpg_data)
        self.current_image = self.image_processor.preprocess(obs)

    def compute_action(self, observation):
        action = self.model.predict(observation)

        return self.action_processor.action(action.astype(float))

    def on_received_get_commands(self, context: Context):
        pwm_left, pwm_right = self.compute_action(self.current_image)
        pwm_left = float(np.clip(pwm_left, -1, +1))
        pwm_right = float(np.clip(pwm_right, -1, +1))
        grey = RGB(0.0, 0.0, 0.0)
        led_commands = LEDSCommands(grey, grey, grey, grey, grey)
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = Duckiebot1Commands(pwm_commands, led_commands)
        context.write('commands', commands)

    def finish(self, context: Context):
        context.info('finish()')


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """ Reads JPG bytes as RGB"""
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data

def main():
    node = PytorchRLBaseline()
    protocol = protocol_agent_duckiebot1
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
