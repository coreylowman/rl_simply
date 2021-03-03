import math
from datetime import datetime

import torch
from torch import nn, optim
import torch.nn.functional as F

import gym
from gym.spaces import Box, Discrete

from memory import ReplayBuffer, Batch
from wrappers import TorchWrapper


def Actor(inp_dim: int, out_dim: int, hid_dim: int = 64):
    return nn.Sequential(
        nn.Linear(inp_dim, hid_dim),
        nn.ReLU(),
        nn.Linear(hid_dim, hid_dim),
        nn.ReLU(),
        nn.Linear(hid_dim, out_dim),
    )


class DQN:
    def __init__(
        self,
        state_space: Box,
        action_space: Discrete,
        mini_batch_size: int = 128,
        replay_buffer_size: int = 50_000,
        learning_starts: int = 1000,
        actor_lr: float = 1e-3,
        discount: float = 0.99,
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.mini_batch_size = mini_batch_size
        self.replay_buffer_size = replay_buffer_size
        self.learning_starts = learning_starts
        self.actor_lr = actor_lr
        self.discount = discount

        self.q = Actor(self.state_space.shape[0], self.action_space.n)
        self.q_target = Actor(self.state_space.shape[0], self.action_space.n)
        self.q_opt = optim.Adam(self.q.parameters(), lr=self.actor_lr)

        self.q_target.load_state_dict(self.q.state_dict())

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> int:
        return self.q(state).argmax().item()

    def update(self, buffer: ReplayBuffer):
        batch = buffer.sample(self.mini_batch_size)

        with torch.no_grad():
            next_action = self.q(batch.next_state).max(dim=1)[1].unsqueeze(1)
            max_q_prime = self.q_target(batch.next_state).gather(dim=1, index=next_action)
            target_q = batch.reward + self.discount * max_q_prime * batch.done

        current_q = self.q(batch.state).gather(dim=1, index=batch.action)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()

    def learn(self, env: gym.Env, eval_env: gym.Env, steps: int):
        buffer = ReplayBuffer(self.state_space, self.action_space, self.replay_buffer_size)

        state, start = env.reset(), datetime.now()
        for i_step in range(steps):
            if i_step < self.learning_starts:
                action = self.action_space.sample()
            else:
                action = self.act(state)

            next_state, reward, done, info = env.step(action)
            buffer.add(state, action, reward, done, next_state)
            state = env.reset() if done else next_state

            if i_step >= self.learning_starts:
                self.update(buffer)

            if i_step >= self.learning_starts and i_step % 100 == 0:
                print(i_step, evaluate(eval_env, 42, self, 5), datetime.now() - start)

            if i_step % 100 == 0:
                self.q_target.load_state_dict(self.q.state_dict())


def evaluate(env: gym.Env, seed: int, agent: DQN, num_episodes: int, render: bool = False) -> float:
    score = 0
    for i_episode in range(num_episodes):
        env.seed(seed + i_episode)
        state, done = env.reset(), False
        while not done:
            state, reward, done, info = env.step(agent.act(state))
            score += reward
    return score / num_episodes


def main(seed=0):
    torch.manual_seed(seed)

    env = TorchWrapper(gym.make("CartPole-v1"))
    env.seed(seed)
    env.action_space.seed(seed)

    eval_env = TorchWrapper(gym.make("CartPole-v1"))
    eval_env.seed(seed + 1)

    agent = DQN(env.observation_space, env.action_space)
    agent.learn(env, eval_env, 20_000)

    print(evaluate(env, seed + 2, agent, 50, True))

    env.close()


if __name__ == "__main__":
    main()
