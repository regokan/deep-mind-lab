"""REINFORCE policy."""

import numpy as np
import torch

from .base import BasePolicy


class ReinforceParallelPolicy(BasePolicy):
    def __init__(self, model, device="cpu"):
        self.device = device

        self.model = model.to(device)
        self.saved_log_probs = []

    # convert states to probability, passing through the policy
    def states_to_prob(self, states):
        states = torch.stack(states)
        policy_input = states.view(-1, *states.shape[-3:])
        return self.model.forward(policy_input.to(self.device)).view(states.shape[:-3])

    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        return self.model(state).cpu()

    def update_exploration(self, R):
        pass

    def update_policy(
        self, rewards, old_probs, states, actions, beta=0.01, action1=None, action2=None
    ):
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
