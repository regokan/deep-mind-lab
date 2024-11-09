"""Base class for policies."""

from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """Abstract base class for policies."""

    def __init__(self, q_table, action_space):
        self.q_table = q_table
        self.action_space = action_space

    @abstractmethod
    def select_action(self, state):
        """Select an action based on the current state."""

    @abstractmethod
    def update_exploration(self, episode):
        """Optional method to update exploration rate (e.g., for GLIE)."""


class ValueBasedPolicy(BasePolicy):
    """Base class for value-based policies like Q-learning, SARSA."""

    def __init__(self, q_table, action_space):
        super().__init__(q_table, action_space)

    @abstractmethod
    def update_policy(self, state, action, reward, next_state, done):
        """Update the policy using a value-based method."""


class MonteCarloPolicy(BasePolicy):
    """Base class for Monte Carlo methods."""

    def __init__(self, q_table, action_space):
        super().__init__(q_table, action_space)

    @abstractmethod
    def update_mc(self, episode_memory):
        """Update the policy using Monte Carlo methods."""
