import torch
from torch import nn, optim
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from memory import ReplayBuffer
from wrappers import TorchWrapper


MINI_BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 5e-4
REPLAY_BUFFER_SIZE = 50_000
TARGET_RESET_INTERVAL = 500
TRAINING_STARTS = 1000
TRAINING_FREQ = 4
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DURATION = 10_000


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
        self.epsilon = EPSILON_START
        self.num_trains = 0

    def explore(self, state: torch.Tensor) -> int:
        if torch.rand(1) < self.epsilon:
            return self.action_space.sample()
        else:
            return self.exploit(state)

    def exploit(self, state: torch.Tensor) -> int:
        return self.q(state).argmax().item()

    def train(self, buffer: ReplayBuffer):
        s, a, r, done, s_prime = buffer.sample(MINI_BATCH_SIZE)

        with torch.no_grad():
            max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
            target_q = r + GAMMA * max_q_prime * done

        current_q = self.q(s).gather(1, a)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.num_trains += 1
        if self.num_trains > 0 and self.num_trains % TARGET_RESET_INTERVAL == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        self.epsilon = max(
            EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * self.num_trains / EPSILON_DURATION
        )


def main():
    import gym

    env = TorchWrapper(gym.make("CartPole-v1"))

    ddqn = DoubleDQN(env.observation_space, env.action_space)
    buffer = ReplayBuffer(env.observation_space, env.action_space, REPLAY_BUFFER_SIZE)

    state, done = env.reset(), False
    episode_rewards = [0]
    while sum(episode_rewards[-100:]) / 100 < 195:
        action = ddqn.explore(state)

        state_prime, reward, done, info = env.step(action)
        buffer.add(state, action, reward, done, state_prime)
        state = state_prime
        episode_rewards[-1] += reward

        if done:
            state, done = env.reset(), False
            episode_rewards.append(0)

        if buffer.size > TRAINING_STARTS and buffer.num_steps % TRAINING_FREQ == 0:
            ddqn.train(buffer)

        if buffer.num_steps > 0 and buffer.num_steps % 1000 == 0:
            print(f"{buffer.num_steps}: {sum(episode_rewards[-100:]) / 100:0.2f}")

    print(f"{buffer.num_steps}: {sum(episode_rewards[-100:]) / 100:0.2f}")

    while True:
        state, done = env.reset(), False
        env.render()
        score = 0
        while not done:
            state, reward, done, info = env.step(ddqn.exploit(state))
            env.render()
            score += reward
        print(score)

    env.close()


if __name__ == "__main__":
    main()
