import copy
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
from torch import optim
from utils import seed, evaluate_policy, ReplayBuffer

from duckietown_rl.teacher import PurePursuitExpert
from wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from env import launch_env
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def warmup(policy, args, env, file_name):
    if not args.do_warmup:
        return policy
    # Lifted from https://github.com/duckietown/gym-duckietown/blob/master/learning/imitation/basic/train_imitation.py

    # Create an imperfect demonstrator
    expert = PurePursuitExpert(env=env)
    replay_buffer = ReplayBuffer(np.inf)

    # let's collect our samples
    for episode in range(0, args.warmup_episodes):
        print("Starting episode", episode)
        obs = env.reset()
        done = False
        for steps in range(0, args.warmup_steps):
            if done:
                break

            # use our 'expert' to predict the next action.
            action = expert.predict(None)
            new_obs, reward, done, info = env.step(action)

            replay_buffer.add(obs, new_obs, action, reward, done)
            obs = new_obs

    actor = policy.actor
    # weight_decay is L2 regularization, helps avoid overfitting
    actor_optimizer = optim.SGD(
        actor.parameters(),
        lr=0.0004,
        weight_decay=1e-3
    )

    critic = policy.critic
    critic_target = policy.critic_target
    critic_optimizer = torch.optim.Adam(critic.parameters())

    """
    def all_warmup_batches(n=args.warmup_batch_size):
        l = len(observations)

        indices = []

        for ndx in range(0, l, n):
            indices.append(list(range(ndx,min(ndx + n, l))))

        random.shuffle(indices)

        for index in indices:
            index = np.array(index)
            yield observations[index], actions[index], rewards[index], dones[index]
    """

    #avg_loss = 0
    for epoch in range(args.warmup_epochs):
        print("Epoch:", epoch)
        for state, next_state, action, reward, done in replay_buffer.all_batches(args.warmup_batch_size):
            state = torch.from_numpy(state).float().to(device)
            action = torch.from_numpy(action).float().to(device)
            next_state = torch.from_numpy(next_state).float().to(device)
            done = torch.from_numpy(done).float().to(device)
            reward = torch.from_numpy(reward).float().to(device)

            model_actions = actor(state)

            actor_optimizer.zero_grad()
            loss = (model_actions - action).norm(2).mean()
            loss.backward()
            actor_optimizer.step()

            target_Q = critic_target(next_state, actor(next_state))
            target_Q = reward + (done * args.discount * target_Q).detach()
            current_Q = critic(state, action)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            # Update the frozen target models
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    policy.actor_target = copy.deepcopy(actor)
    policy.save(file_name, directory="./pytorch_models")
    return policy
