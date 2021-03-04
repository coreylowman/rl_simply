from datetime import datetime

import torch
from torch import nn, optim
import torch.nn.functional as F

import gym
from gym.spaces import Box

from memory import ReplayBuffer, Batch
from wrappers import TorchWrapper, NormalizeActionsWrapper


def MLP(inp_dim: int, out_dim: int, *layers: nn.Module, hid_dim: int = 256):
    return nn.Sequential(
        nn.Linear(inp_dim, hid_dim),
        nn.ReLU(),
        nn.Linear(hid_dim, hid_dim),
        nn.ReLU(),
        nn.Linear(hid_dim, out_dim),
        *layers,
    )


def TanhActor(inp_dim: int, out_dim: int, *args, **kwargs):
    return MLP(inp_dim, out_dim, *args, nn.Tanh(), **kwargs)


def Critic(inp_dim: int, out_dim: int, *args, **kwargs):
    return MLP(inp_dim + out_dim, 1, *args, **kwargs)


class TwinCritics(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.a = Critic(*args, **kwargs)
        self.b = Critic(*args, **kwargs)

    def forward(self, x):
        return self.a(x), self.b(x)


class TD3:
    mini_batch_size: int = 256
    replay_buffer_size: int = 50_000
    learning_starts: int = 1000
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    rollout_noise: float = 0.2
    train_noise: float = 0.2
    noise_clip: float = 0.5
    delay: int = 2

    def __init__(self, state_space: Box, action_space: Box):
        self.state_space = state_space
        self.action_space = action_space

        self.actor = TanhActor(state_space.shape[0], action_space.shape[0])
        self.actor_target = TanhActor(state_space.shape[0], action_space.shape[0])
        self.critics = TwinCritics(state_space.shape[0], action_space.shape[0])
        self.critic_targets = TwinCritics(state_space.shape[0], action_space.shape[0])

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critics_opt = optim.Adam(self.critics.parameters(), lr=self.critic_lr)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_targets.load_state_dict(self.critics.state_dict())

        self.num_updates = 1

    @torch.no_grad()
    def act(self, state: torch.Tensor, is_training: bool = False) -> int:
        action = self.actor(state)
        if is_training:
            action += self.rollout_noise * torch.randn_like(action)
            action = torch.clamp(action, -1, 1)
        return action

    def update(self, batch: Batch):
        self.update_critics(batch)
        if self.num_updates % self.delay == 0:
            self.update_actor(batch)
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critics, self.critic_targets)
        self.num_updates += 1

    def update_critics(self, batch: Batch):
        with torch.no_grad():
            next_action = self.actor_target(batch.next_state)
            next_action += self.train_noise * torch.randn_like(next_action).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = next_action.clamp(-1, 1)

            next_q1, next_q2 = self.critic_targets(torch.cat((batch.next_state, next_action), -1))
            next_q = torch.minimum(next_q1, next_q2)
            target_q = batch.reward + self.discount * batch.done * next_q

        q1, q2 = self.critics(torch.cat((batch.state, batch.action), -1))
        critic_loss = F.smooth_l1_loss(q1, target_q) + F.smooth_l1_loss(q2, target_q)

        self.critics_opt.zero_grad()
        critic_loss.backward()
        self.critics_opt.step()

    def update_actor(self, batch: Batch):
        action = self.actor(batch.state)

        actor_loss = -self.critics.a(torch.cat((batch.state, action), -1)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

    def soft_update(self, m, target_m):
        for p, target_p in zip(m.parameters(), target_m.parameters()):
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


def evaluate(env: gym.Env, seed: int, td3: TD3, num_episodes: int, render: bool = False) -> float:
    score = 0
    for i_eval_eps in range(num_episodes):
        env.seed(seed + i_eval_eps)
        state, done = env.reset(), False
        while not done:
            state, reward, done, info = env.step(td3.act(state))
            score += reward
    return score.item() / num_episodes


def main(seed=0):
    torch.manual_seed(seed)

    env = TorchWrapper(NormalizeActionsWrapper(gym.make("Pendulum-v0")))
    env.seed(seed)
    env.action_space.seed(seed)

    eval_env = TorchWrapper(NormalizeActionsWrapper(gym.make("Pendulum-v0")))
    eval_env.seed(seed + 1)

    td3 = TD3(env.observation_space, env.action_space)
    td3.learn(env, eval_env, 10_000)

    print(evaluate(env, seed + 2, td3, 50, True))

    env.close()


if __name__ == "__main__":
    main()
