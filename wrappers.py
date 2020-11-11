import torch
from gym.core import ObservationWrapper, ActionWrapper


class TorchWrapper(ObservationWrapper, ActionWrapper):
    def action(self, action):
        return action.detach().cpu().numpy()

    def reverse_action(self, action):
        return torch.from_numpy(action).float()

    def observation(self, observation):
        return torch.from_numpy(observation).float()