"""Base Q-table class."""

from abc import ABC, abstractmethod


class BaseQTable(ABC):
    """Abstract class for Q-table storage and operations."""

    @abstractmethod
    def get_q_value(self, state, action):
        """Get the Q-value for a given state and action."""

    @abstractmethod
    def set_q_value(self, state, action, value):
        """Set the Q-value for a given state and action."""

    @abstractmethod
    def initialize(self, state_space, action_space):
        """Initialize the Q-table for given state and action spaces."""
