import gym

from wrappers import (
    NormalizeWrapper,
    ImgWrapper,
    DtRewardWrapper,
    ActionWrapper,
    ResizeWrapper,
    SteeringToWheelVelWrapper,
)


def launch_env(id=None):
    env = None
    if id is None:
        from gym_duckietown.simulator import Simulator

        env = Simulator(
            seed=123,  # random seed
            map_name="loop_empty",
            max_steps=500001,  # we don't want the gym to reset itself
            domain_rand=False,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            distortion=True,
        )
    else:
        env = gym.make(id)

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    env = SteeringToWheelVelWrapper(env)
    env = ActionWrapper(env)
    # env = DtRewardWrapper(env)

    return env
