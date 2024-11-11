"""Cross Entropy policy."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BasePolicy


class CrossEntropyPolicy(BasePolicy, nn.Module):
    def __init__(self, s_size, a_size, h_size=16):
        super().__init__()
        self.s_size = s_size
        self.a_size = a_size
        self.h_size = h_size
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)

    def _forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.tanh(self.fc2(x))
        return x.cpu().data

    def select_action(self, state):
        action = self._forward(state)
        return action

    def get_weights_dim(self):
        return (self.s_size + 1) * self.h_size + (self.h_size + 1) * self.a_size

    def update_exploration(self, weights):
        # separate the weights for each layer
        fc1_end = (self.s_size * self.h_size) + self.h_size
        fc1_W = torch.from_numpy(
            weights[: self.s_size * self.h_size].reshape(self.s_size, self.h_size)
        )
        fc1_b = torch.from_numpy(weights[self.s_size * self.h_size : fc1_end])
        fc2_W = torch.from_numpy(
            weights[fc1_end : fc1_end + (self.h_size * self.a_size)].reshape(
                self.h_size, self.a_size
            )
        )
        fc2_b = torch.from_numpy(weights[fc1_end + (self.h_size * self.a_size) :])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
