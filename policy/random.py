"""Random policy."""

from .base import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, action_space):
        """
        Initializes the RandomPolicy with the given action space.

        Parameters:
        action_space: Action space of the environment.
        """
        self.action_space = action_space

    def select_action(self, _):
        """Selects a random action, ignoring the current state."""
        return self.action_space.sample()

    def update_exploration(self, episode):
        """Does nothing since this policy doesn't learn from experience."""
