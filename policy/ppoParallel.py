"""REINFORCE policy."""

import numpy as np
import torch
import torch.nn as nn

from .reinforceParallel import ReinforceParallelPolicy


class PPOParallelPolicy(ReinforceParallelPolicy):
    """
    Implementation of PPO policy with entropy regularization.

    This policy works with environments in which agents have to decide between two actions.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize the policy.

        Args:
            model: PyTorch model.
            device: Device to use for computation.
        """
        super().__init__(model, device)

    def update_policy(
        self,
        rewards,
        old_probs,
        states,
        actions,
        beta=0.01,
        action1=None,
        action2=None,
        epsilon=0.2,
    ):
        """
        Update the policy using the REINFORCE algorithm.

        Args:
            rewards: Rewards for the episode.
            old_probs: Old probabilities for the actions.
            states: States for the episode.
            actions: Actions taken in the episode.
            beta: Entropy regularization parameter.
            action1: First action.
            action2: Second action.
        """
        actions = torch.from_numpy(np.array(actions, dtype=np.int8)).to(self.device)
        old_probs = torch.from_numpy(np.array(old_probs, dtype=np.float32)).to(
            self.device
        )
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(self.device)

        # convert states to policy (or probability)
        new_probs = self.states_to_prob(states)
        action1 = torch.tensor(action1, device=self.device)
        new_probs = torch.where(
            (actions.eq(action1)).type(torch.bool), new_probs, 1.0 - new_probs
        )

        ratio = new_probs / old_probs

        clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        clipped_surrogate = torch.min(ratio * rewards, clip * rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(
            new_probs * torch.log(old_probs + 1.0e-10)
            + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.0e-10)
        )

        L = -torch.mean(clipped_surrogate + beta * entropy)
        L.backward()

        del L
