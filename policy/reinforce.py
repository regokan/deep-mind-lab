"""REINFORCE policy."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Categorical

from .base import BasePolicy


class ReinforcePolicy(BasePolicy, nn.Module):
    """
    REINFORCE policy.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize the REINFORCE policy.

        Args:
            model: The model to use for the policy.
            device: The device to use for the policy.
        """
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.saved_log_probs = []

    def select_action(self, state):
        """
        Select an action from the policy.

        Args:
            state: The state to select an action from.
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        state = state.to(self.device)
        probs = self.model.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action

    def update_exploration(self, R):
        """
        Update the exploration of the policy.

        Args:
            R: The reward to update the exploration with.
        """
        policy_loss = []
        for log_prob in self.saved_log_probs:
            if isinstance(R, np.ndarray):
                R = torch.from_numpy(R)
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.saved_log_probs = []
