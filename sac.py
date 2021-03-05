from datetime import datetime

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform

import gym
from gym.spaces import Box

from memory import ReplayBuffer, Batch
from wrappers import TorchWrapper, NormalizeActionsWrapper


def MLP(inp_dim: int, out_dim: int, hid_dim: int = 256):
    return nn.Sequential(
        nn.Linear(inp_dim, hid_dim),
        nn.ReLU(),
        nn.Linear(hid_dim, hid_dim),
        nn.ReLU(),
        nn.Linear(hid_dim, out_dim),
    )


def Critic(inp_dim: int, out_dim: int, *args, **kwargs):
    return MLP(inp_dim + out_dim, 1, **kwargs)


class TwinCritics(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.a = Critic(*args, **kwargs)
        self.b = Critic(*args, **kwargs)

    def forward(self, x):
        return self.a(x), self.b(x)


class TanhGaussianActor(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, *args, **kwargs):
        super().__init__()
        self.trunk = MLP(inp_dim, 2 * out_dim, *args, **kwargs)

    def forward(self, x, sample=True):
        mean, log_std = self.trunk(x).chunk(2, -1)
        log_std = torch.clamp(log_std, -20, 2)

        dist = Normal(mean, torch.exp(log_std) * float(sample))
        dist = TransformedDistribution(dist, TanhTransform())

        action = dist.rsample()
        log_prob = dist.log_prob(action)

        return action, log_prob


class SAC:
    mini_batch_size: int = 256
    replay_buffer_size: int = 50_000
    learning_starts: int = 1000
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005

    def __init__(self, state_space: Box, action_space: Box):
        self.state_space = state_space
        self.action_space = action_space

        self.actor = TanhGaussianActor(state_space.shape[0], action_space.shape[0])
        self.critics = TwinCritics(state_space.shape[0], action_space.shape[0])
        self.critic_targets = TwinCritics(state_space.shape[0], action_space.shape[0])
        self.log_alpha = nn.Parameter(torch.tensor([0.0]))

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critics_opt = optim.Adam(self.critics.parameters(), lr=self.critic_lr)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.alpha_lr)

        self.critic_targets.load_state_dict(self.critics.state_dict())

        self.target_entropy = -torch.prod(torch.tensor(action_space.shape))

    @torch.no_grad()
    def act(self, state: torch.Tensor, is_training: bool = False) -> int:
        action, _log_prob = self.actor(state, sample=is_training)
        return action

    def update(self, batch: Batch):
        self.update_critics(batch)
        self.update_actor(batch)
        self.update_alpha(batch)
        self.soft_update_target_critics()

    def update_critics(self, batch: Batch):
        with torch.no_grad():
            alpha = torch.exp(self.log_alpha)
            next_action, next_log_prob = self.actor(batch.next_state)

            next_q1, next_q2 = self.critic_targets(torch.cat((batch.next_state, next_action), -1))
            next_q = torch.minimum(next_q1, next_q2)
            target_q = batch.reward + self.discount * batch.done * (next_q - alpha * next_log_prob)

        q1, q2 = self.critics(torch.cat((batch.state, batch.action), -1))
        critic_loss = F.smooth_l1_loss(q1, target_q) + F.smooth_l1_loss(q2, target_q)

        self.critics_opt.zero_grad()
        critic_loss.backward()
        self.critics_opt.step()

    def update_actor(self, batch: Batch):
        with torch.no_grad():
            alpha = torch.exp(self.log_alpha)

        action, log_prob = self.actor(batch.state)

        q1, q2 = self.critics(torch.cat((batch.state, action), -1))
        q = torch.minimum(q1, q2)

        actor_loss = torch.mean(log_prob * alpha - q)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

    def update_alpha(self, batch: Batch):
        with torch.no_grad():
            _, log_prob = self.actor(batch.state)

        alpha_loss = -torch.mean(self.log_alpha * (log_prob + self.target_entropy))

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

    def soft_update_target_critics(self):
        for p, target_p in zip(self.critics.parameters(), self.critic_targets.parameters()):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)

    def learn(self, env: gym.Env, eval_env: gym.Env, steps: int):
        buffer = ReplayBuffer(env.observation_space, env.action_space, self.replay_buffer_size)

        state, start = env.reset(), datetime.now()
        for i_step in range(steps):
            if i_step < self.learning_starts:
                action = torch.from_numpy(self.action_space.sample()).float()
            else:
                action = self.act(state, is_training=True)

            next_state, reward, done, info = env.step(action)
            buffer.add(state, action, reward, done, next_state)
            state = env.reset() if done else next_state

            if i_step >= self.learning_starts:
                self.update(buffer.sample(self.mini_batch_size))

            if i_step % 1000 == 0 and i_step > 0:
                print(i_step, evaluate(eval_env, 42, self, 5), datetime.now() - start)


def evaluate(env: gym.Env, seed: int, sac: SAC, num_episodes: int, render: bool = False) -> float:
    score = 0
    for i_eval_eps in range(num_episodes):
        env.seed(seed + i_eval_eps)
        state, done = env.reset(), False
        while not done:
            state, reward, done, info = env.step(sac.act(state))
            score += reward
    return score.item() / num_episodes


def main(seed=0):
    torch.manual_seed(seed)

    env = TorchWrapper(NormalizeActionsWrapper(gym.make("Pendulum-v0")))
    env.seed(seed)
    env.action_space.seed(seed)

    eval_env = TorchWrapper(NormalizeActionsWrapper(gym.make("Pendulum-v0")))
    eval_env.seed(seed + 1)

    sac = SAC(env.observation_space, env.action_space)
    sac.learn(env, eval_env, 10_000)

    print(evaluate(env, seed + 2, sac, 50, True))

    env.close()


if __name__ == "__main__":
    main()
