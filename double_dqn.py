import torch
from torch import nn, optim
import torch.nn.functional as F
import gym
from gym.spaces import Box, Discrete
from memory import ReplayBuffer
from wrappers import TorchWrapper

MINI_BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 5e-4
REPLAY_BUFFER_SIZE = 50_000
TARGET_UPDATE_DELAY = 1000
NUM_RANDOM_ACTIONS = 2000
LEARNING_STARTS = 1000
UPDATES_PER_STEP = 4
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DURATION = 5_000


def make(state_space: Box, action_space: Discrete, activation_fn=nn.ReLU):
    return nn.Sequential(
        nn.Linear(state_space.shape[0], 32),
        activation_fn(),
        nn.Linear(32, 32),
        activation_fn(),
        nn.Linear(32, action_space.n),
    )


class DoubleDQN:
    def __init__(self, state_space: Box, action_space: Discrete):
        self.state_space = state_space
        self.action_space = action_space

        self.q = make(state_space, action_space)
        self.q_target = make(state_space, action_space)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=LEARNING_RATE)

        self.num_steps = 0
        self.num_updates = 0

    @property
    def epsilon(self):
        return max(
            EPSILON_END,
            EPSILON_START - (EPSILON_START - EPSILON_END) * self.num_steps / EPSILON_DURATION,
        )

    @torch.no_grad()
    def act(self, state: torch.Tensor, is_training: bool = False) -> int:
        if is_training:
            self.num_steps += 1

            if self.num_steps < NUM_RANDOM_ACTIONS:
                return self.action_space.sample()

            if torch.rand(1) < self.epsilon:
                return self.action_space.sample()

        return self.q(state.unsqueeze(0)).argmax().item()

    def update(self, buffer: ReplayBuffer):
        if buffer.size < LEARNING_STARTS:
            return

        for i_update in range(UPDATES_PER_STEP):
            self._update_once(buffer)

        self.num_updates += 1
        if self.num_updates > 0 and self.num_updates % TARGET_UPDATE_DELAY == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    def _update_once(self, buffer: ReplayBuffer):
        state, action, reward, done, next_state = buffer.sample(MINI_BATCH_SIZE)

        with torch.no_grad():
            max_q_prime = self.q_target(next_state).max(dim=1)[0].unsqueeze(1)
            target_q = reward + GAMMA * max_q_prime * done

        current_q = self.q(state).gather(dim=1, index=action)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self, env: gym.Env, buffer: ReplayBuffer, steps: int):
        state = env.reset()

        episode_reward = 0
        for i_step in range(steps):
            action = self.act(state, is_training=True)

            next_state, reward, done, info = env.step(action)
            # env.render()

            self.update(buffer.add(state, action, reward, done, next_state))

            state = next_state
            episode_reward += reward
            if done:
                print(self.num_steps, self.epsilon, episode_reward)

                state = env.reset()
                episode_reward = 0


def main(seed=0):
    torch.manual_seed(seed)

    env = TorchWrapper(gym.make("CartPole-v1"))
    env.seed(seed)

    ddqn = DoubleDQN(env.observation_space, env.action_space)
    buffer = ReplayBuffer(env.observation_space, env.action_space, REPLAY_BUFFER_SIZE)

    ddqn.learn(env, buffer, 50_000)

    torch.save(ddqn.q.state_dict(), "ddqn.pt")
    ddqn.q.load_state_dict(torch.load("ddqn.pt"))

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
