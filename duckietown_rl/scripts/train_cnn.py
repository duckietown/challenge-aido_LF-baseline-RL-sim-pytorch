import pickle
import random
import resource
import gym_duckietown
import numpy as np
import torch
import gym
import os

from args import get_ddpg_args_train
from ddpg import DDPG
from utils import seed, evaluate_policy, ReplayBuffer
from wrappers import (
    NormalizeWrapper,
    ImgWrapper,
    DtRewardWrapper,
    ActionWrapper,
    ResizeWrapper,
    SteeringToWheelVelWrapper,
)
from env import launch_env

policy_name = "DDPG"

print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_ddpg_args_train()

if args.log_file != None:
    print('You asked for a log file. "Tee-ing" print to also print to file "' + args.log_file + '" now...')

    import subprocess, os, sys

    tee = subprocess.Popen(["tee", args.log_file], stdin=subprocess.PIPE)
    # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
    # of any child processes we spawn)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


file_name = "{}_{}".format(
    policy_name,
    str(args.seed),
)

if not os.path.exists("./results"):
    os.makedirs("./results")
if args.save_models and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

env = launch_env()

# Set seeds
seed(args.seed)

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])


# Initialize policy
policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")

replay_buffer = ReplayBuffer(args.replay_buffer_max_size)

# Evaluate untrained policy
evaluations = [evaluate_policy(env, policy)]

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
episode_reward = None
env_counter = 0

while total_timesteps < args.max_timesteps:

    if done:
        print(f"Done @ {total_timesteps}")

        if total_timesteps != 0:
            print("Replay buffer length is ", len(replay_buffer.storage))
            print(
                ("Total T: %d Episode Num: %d Episode T: %d Reward: %f")
                % (total_timesteps, episode_num, episode_timesteps, episode_reward)
            )
            policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            evaluations.append(evaluate_policy(env, policy))

            if args.save_models:
                policy.save(file_name, directory="./pytorch_models")
            np.savez("./results/{}.npz".format(file_name), evaluations)

        # Reset environment
        env_counter += 1
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Select action randomly or according to policy
    if total_timesteps < args.start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.predict(np.array(obs))
        if args.expl_noise != 0:
            action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                env.action_space.low, env.action_space.high
            )

    # Perform action
    new_obs, reward, done, _ = env.step(action)
    if (
        action[0] < 0.001
    ):  # Penalise slow actions: helps the bot to figure out that going straight > turning in circles
        reward = 0

    if episode_timesteps >= args.env_timesteps:
        done = True

    done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
    episode_reward += reward

    # Store data in replay buffer
    replay_buffer.add(obs, new_obs, action, reward, done_bool)
    # approximate_size(replay_buffer)   #TODO rm

    obs = new_obs

    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

# Final evaluation
evaluations.append(evaluate_policy(env, policy))

if args.save_models:
    policy.save(file_name, directory="./pytorch_models")
np.savez("./results/{}.npz".format(file_name), evaluations)
