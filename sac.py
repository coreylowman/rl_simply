import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gym
from gym.spaces import Box, Discrete

from memory import ReplayBuffer, Batch
from wrappers import TorchWrapper, NormalizeActionsWrapper

MINI_BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 50_000
LEARNING_STARTS = 1000
UPDATES_PER_STEP = 1

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4
DISCOUNT = 0.99
TAU = 0.005
TEMPERATURE = 1.0


def MLP(inp_dim: int, out_dim: int, hid_dim: int = 256, activation_fn=nn.ReLU):
    return nn.Sequential(
        nn.Linear(inp_dim, hid_dim),
        activation_fn(),
        nn.Linear(hid_dim, hid_dim),
        activation_fn(),
        nn.Linear(hid_dim, out_dim),
    )


def Critic(inp_dim: int, out_dim: int, *args, **kwargs):
    return MLP(inp_dim + out_dim, 1, **kwargs)


class ModulePair(nn.Module):
    def __init__(self, cls, *args, **kwargs):
        super().__init__()

        self.a = cls(*args, **kwargs)
        self.b = cls(*args, **kwargs)

    def forward(self, x):
        return self.a(x), self.b(x)


class Actor(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, *args, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.trunk = MLP(inp_dim, 2 * out_dim, *args, **kwargs)

    def forward(self, x, temperature=TEMPERATURE):
        x = self.trunk(x)

        mean, logstd = torch.split(x, self.out_dim, -1)
        logstd = torch.clamp(logstd, -20, 2)

        dist = Normal(mean, torch.exp(logstd) * temperature)

        action = dist.rsample()
        squashed_action = torch.tanh(action)

        log_prob = dist.log_prob(action)
        squash_correction = torch.sum(
            torch.log(1 - squashed_action.square() + 1e-6), -1, keepdim=True
        )
        log_prob -= squash_correction

        return squashed_action, log_prob


class SAC:
    def __init__(self, state_space: Box, action_space: Box):
        self.state_space = state_space
        self.action_space = action_space

        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]

        self.actor = Actor(state_dim, action_dim)
        self.critics = ModulePair(Critic, state_dim, action_dim)
        self.critic_targets = ModulePair(Critic, state_dim, action_dim)
        self.log_alpha = nn.Parameter(torch.tensor([0.0]))

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critics_opt = optim.Adam(self.critics.parameters(), lr=CRITIC_LR)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=ALPHA_LR)

        self.critic_targets.load_state_dict(self.critics.state_dict())

        self.target_entropy = -np.prod(action_space.shape)

    @torch.no_grad()
    def act(self, state: torch.Tensor, is_training: bool = False) -> int:
        if is_training:
            action, _ = self.actor(state)
            return action

        action, _ = self.actor(state, temperature=0.0)
        return action

    def update(self, buffer: ReplayBuffer):
        batch = buffer.sample(MINI_BATCH_SIZE)

        critic_loss = self.update_critics(batch)
        actor_loss = self.update_actor(batch)
        alpha_loss = self.update_alpha(batch)

        self.soft_update_target_critics()

        # print(critic_loss, actor_loss, alpha_loss)

    def update_actor(self, batch: Batch):
        with torch.no_grad():
            alpha = torch.exp(self.log_alpha)

        action, log_prob = self.actor(batch.state)

        q1, q2 = self.critics(torch.cat((batch.state, action), -1))
        q = torch.minimum(q1, q2)

        actor_loss = torch.mean(log_prob * alpha.detach() - q)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return actor_loss.item()

    def update_critics(self, batch: Batch):
        with torch.no_grad():
            alpha = torch.exp(self.log_alpha)
            next_action, next_log_prob = self.actor(batch.next_state)

            next_q1, next_q2 = self.critic_targets(torch.cat((batch.next_state, next_action), -1))
            next_q = torch.minimum(next_q1, next_q2)
            target_q = batch.reward + DISCOUNT * batch.done * (next_q - alpha * next_log_prob)

        q1, q2 = self.critics(torch.cat((batch.state, batch.action), -1))
        q1_loss = 0.5 * (q1 - target_q.detach()).square().mean()
        q2_loss = 0.5 * (q2 - target_q.detach()).square().mean()
        critic_loss = q1_loss + q2_loss

        self.critics_opt.zero_grad()
        critic_loss.backward()
        self.critics_opt.step()

        return critic_loss.item()

    def soft_update_target_critics(self):
        curr_state = self.critics.state_dict()
        targ_state = self.critic_targets.state_dict()
        self.critic_targets.load_state_dict(
            {k: TAU * curr_state[k] + (1.0 - TAU) * targ_state[k] for k in curr_state}
        )

    def update_alpha(self, batch: Batch):
        with torch.no_grad():
            _, log_prob = self.actor(batch.state)

        alpha_loss = torch.mean(-1.0 * self.log_alpha * (log_prob + self.target_entropy).detach())

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        return alpha_loss.item()

    def learn(self, env: gym.Env, eval_env: gym.Env, buffer: ReplayBuffer, steps: int):
        state = env.reset()
        for i_step in range(steps):
            if i_step < LEARNING_STARTS:
                action = torch.from_numpy(self.action_space.sample()).float()
            else:
                action = self.act(state, is_training=True)

            next_state, reward, done, info = env.step(action)

            buffer.add(state, action, reward, done, next_state)

            if i_step >= LEARNING_STARTS:
                for i_update in range(UPDATES_PER_STEP):
                    self.update(buffer)

            state = next_state
            if done:
                state = env.reset()

            if i_step % 1000 == 0:
                eval_env.seed(42)
                score = 0
                for i_eval_eps in range(1):
                    state, done = eval_env.reset(), False
                    eval_env.render()
                    while not done:
                        action = self.act(state)
                        state, reward, done, info = eval_env.step(action)
                        eval_env.render()
                        score += reward
                print(i_step, torch.exp(self.log_alpha).item(), score.item())


def main(seed=0):
    torch.manual_seed(seed)

    env = TorchWrapper(NormalizeActionsWrapper(gym.make("Pendulum-v0")))
    env.seed(seed)

    eval_env = TorchWrapper(NormalizeActionsWrapper(gym.make("Pendulum-v0")))
    eval_env.seed(seed + 1)

    sac = SAC(env.observation_space, env.action_space)
    buffer = ReplayBuffer(env.observation_space, env.action_space, REPLAY_BUFFER_SIZE)

    sac.learn(env, eval_env, buffer, 50_000)

    state = env.reset()
    env.render()
    episode_reward = 0
    while True:
        state, reward, done, info = env.step(ddqn.act(state))
        episode_reward += reward
        env.render()
        if done:
            state = env.reset()
            print(episode_reward)
            episode_reward = 0

    env.close()


if __name__ == "__main__":
    main()
