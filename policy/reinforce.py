"""REINFORCE policy."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Categorical

from .base import BasePolicy


class ReinforcePolicy(BasePolicy, nn.Module):
    def __init__(self, s_size, a_size, h_size=16, device="cpu"):
        super().__init__()
        self.s_size = s_size
        self.a_size = a_size
        self.h_size = h_size
        self.device = device

        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)
        self.saved_log_probs = []

        # Apply weight initialization
        self._init_weights()

    def _init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)

    def _forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self._forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update_exploration(self, R):
        policy_loss = []
        for log_prob in self.saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.saved_log_probs = []
