"""Random policy."""

import numpy as np

from .base import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, action_space):
        """
        Initializes the RandomPolicy with the given action space.

        Parameters:
        action_space: Action space of the environment.
        """
        self.action_space = action_space

    def select_action(self, observation):
        """Selects a random action, ignoring the current state."""
        # Check if action space is from OpenAI Gym
        if hasattr(self.action_space, "sample"):
            return self.action_space.sample()

        # If using ML-Agents, handle mixed action space
        if hasattr(self.action_space, "is_discrete") or hasattr(
            self.action_space, "is_continuous"
        ):
            return self.action_space.random_action(observation.shape[0])

    def update_exploration(self, episode):
        """Does nothing since this policy doesn't learn from experience."""
