"""Base trainer class."""

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Abstract class for a learning method."""

    def __init__(self, policy, environment):
        self.policy = policy
        self.environment = environment

    @abstractmethod
    def train(self, episodes, **kwargs):
        """Train the policy on the environment for a certain number of episodes."""

    @abstractmethod
    def evaluate(self, episodes, **kwargs):
        """Evaluate the policy on the environment."""
