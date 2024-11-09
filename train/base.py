"""Base trainer class."""

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Abstract class for a learning method."""

    def __init__(self, policy, environment):
        self.policy = policy
        self.environment = environment

    @abstractmethod
    def train(self, episodes):
        """Train the policy on the environment for a certain number of episodes."""

    @abstractmethod
    def evaluate(self, episodes):
        """Evaluate the policy on the environment."""
