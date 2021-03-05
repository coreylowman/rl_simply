from collections import namedtuple

import torch
import numpy as np
from gym.spaces import Space, Box, Discrete

Batch = namedtuple(
    "Batch", ["state", "action", "reward", "done", "next_state", "indices", "weights"]
)


class ReplayBuffer:
    def __init__(self, state_space: Box, action_space: Space, size: int, seed: int = 0):
        self.state = torch.zeros(size, state_space.shape[0], dtype=torch.float)
        if isinstance(action_space, Discrete):
            self.action = torch.zeros(size, 1, dtype=torch.int64)
        elif isinstance(action_space, Box):
            self.action = torch.zeros(size, action_space.shape[0], dtype=torch.float)
        self.reward = torch.zeros(size, 1, dtype=torch.float)
        self.done = torch.zeros(size, 1, dtype=torch.float)
        self.state_prime = torch.zeros(size, state_space.shape[0])

        self.pointer = 0
        self.size = 0
        self.max_size = size

        self.np_random = np.random.RandomState(seed)

    def add(self, state, action, reward, done, state_prime) -> "ReplayBuffer":
        self.state[self.pointer] = state
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.done[self.pointer] = 0.0 if done else 1.0
        self.state_prime[self.pointer] = state_prime

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self

    def sample(self, batch_size):
        indices = torch.from_numpy(self.np_random.choice(self.size, batch_size)).long()
        return Batch(
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.done[indices],
            self.state_prime[indices],
            indices,
            torch.ones_like(indices),
        )


class PrioritizedReplayBuffer:
    def __init__(
        self,
        state_space: Box,
        action_space: Space,
        size: int,
        alpha: float,
        beta: float,
        beta_increment: float,
        seed: int = 0,
    ):
        self.state = torch.zeros(size, state_space.shape[0], dtype=torch.float)
        if isinstance(action_space, Discrete):
            self.action = torch.zeros(size, 1, dtype=torch.int64)
        elif isinstance(action_space, Box):
            self.action = torch.zeros(size, action_space.shape[0], dtype=torch.float)
        self.reward = torch.zeros(size, 1)
        self.done = torch.zeros(size, 1)
        self.state_prime = torch.zeros(size, state_space.shape[0])
        self.priorities = torch.ones(size)

        self.pointer = 0
        self.size = 0
        self.max_size = size

        self.num_steps = 0

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.np_random = np.random.RandomState(seed)

    def add(self, state, action, reward, done, state_prime) -> "ReplayBuffer":
        self.state[self.pointer] = state
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.done[self.pointer] = 0.0 if done else 1.0
        self.state_prime[self.pointer] = state_prime
        self.priorities[self.pointer] = self.priorities.max()

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.num_steps += 1

        return self

    def sample(self, batch_size):
        probs = self.priorities[: self.size].pow(self.alpha)
        probs /= probs.sum()

        indices = torch.from_numpy(
            self.np_random.choice(self.size, batch_size, p=probs.numpy())
        ).long()

        weights = (self.size * probs[indices]).pow(-self.beta)
        weights /= weights.max()

        self.beta = min(self.beta + self.beta_increment, 1)

        return Batch(
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.done[indices],
            self.state_prime[indices],
            indices,
            weights.unsqueeze(1),
        )

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities