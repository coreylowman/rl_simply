from collections import namedtuple

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from gym.spaces import Space, Box, Discrete

Batch = namedtuple("Batch", ["state", "action", "reward", "done", "next_state"])


class ReplayBuffer:
    def __init__(self, state_space: Box, action_space: Space, size: int):
        self.state = torch.zeros(size, state_space.shape[0], dtype=torch.float)
        if isinstance(action_space, Discrete):
            self.action = torch.zeros(size, 1, dtype=torch.int64)
        elif isinstance(action_space, Box):
            self.action = torch.zeros(size, action_space.shape[0], dtype=torch.float)
        self.reward = torch.zeros(size, 1)
        self.done = torch.zeros(size, 1)
        self.state_prime = torch.zeros(size, state_space.shape[0])

        self.pointer = 0
        self.size = 0
        self.max_size = size

        self.num_steps = 0

    def add(self, state, action, reward, done, state_prime) -> "ReplayBuffer":
        self.state[self.pointer] = state
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.done[self.pointer] = 0.0 if done else 1.0
        self.state_prime[self.pointer] = state_prime

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.num_steps += 1

        return self

    def sample(self, batch_size):
        indices = torch.randperm(self.size)[:batch_size]
        return Batch(
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.done[indices],
            self.state_prime[indices],
        )

    def iter_samples(self, batch_size):
        for indices in BatchSampler(
            SubsetRandomSampler(range(self.size)), batch_size, drop_last=True
        ):
            yield (
                self.state[indices],
                self.action[indices],
                self.reward[indices],
                self.done[indices],
                self.state_prime[indices],
            )
