import torch
from gym.spaces import Box
from gym.core import ObservationWrapper, ActionWrapper


class TorchWrapper(ObservationWrapper, ActionWrapper):
    def action(self, action):
        return action.detach().cpu().numpy()

    def reverse_action(self, action):
        return torch.from_numpy(action).float()

    def observation(self, observation):
        return torch.from_numpy(observation).float()


class NormalizeActionsWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Box(-1, 1, shape=env.action_space.shape)

    def action(self, action):
        action = (action + 1) / 2
        action = (
            action * (self.env.action_space.high - self.env.action_space.low)
            + self.env.action_space.low
        )
        return action

    def reverse_action(self, action):
        action = (action - self.env.action_space.low) / (
            self.env.action_space.high - self.env.action_space.low
        )
        action = 2 * action - 1
        return action
