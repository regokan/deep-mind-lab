"""REINFORCE policy."""

import numpy as np
import torch
import torch.nn as nn

from .reinforceParallel import ReinforceParallelPolicy


class PPO_MDA_UE_Policy(ReinforceParallelPolicy):
    """
    Implementation of REINFORCE policy with entropy regularization where:
        - MDA: Multiple Discrete Action
        - UE: Unity Environment
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize the policy.

        Args:
            model: PyTorch model.
            device: Device to use for computation.
        """
        super().__init__(model, device)

    # convert states to probability, passing through the policy
    def states_to_prob(self, states):
        """
        Convert states to probability.

        Args:
            states: The states to convert to probability.
        """
        states = torch.stack(states)
        policy_input = states.view(-1, *states.shape[-3:])
        return self.model.forward(policy_input.to(self.device))

    def update_policy(
        self, rewards, old_probs, states, beta=0.01
    ):
        """
        Update the policy using the REINFORCE algorithm.

        Args:
            rewards: Rewards for the episode.
            old_probs: Old probabilities for the actions.
            states: States for the episode.
            beta: Entropy regularization parameter.
        """
        old_probs = torch.from_numpy(np.array(old_probs, dtype=np.float32)).to(
            self.device
        )
        steps, batch, action_size = old_probs.shape

        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(self.device)
        rewards = rewards.unsqueeze(-1)
        rewards = rewards.expand(-1, -1, action_size)

        new_probs = self.states_to_prob(states)

        new_probs = new_probs.view(steps, batch, action_size)
        ratio = new_probs / old_probs


        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(
            new_probs * torch.log(old_probs + 1.0e-10)
            + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.0e-10)
        )

        L = -torch.mean(ratio * rewards + beta * entropy)
        L.backward()

        del L
