"""Dictionary Q-table."""

from collections import defaultdict

import numpy as np

from .base import BaseQTable


class DictionaryQTable(BaseQTable):
    """Concrete Q-table using a dictionary to store Q-values."""

    def __init__(self):
        self.q_values = {}

    def initialize(self, action_space):
        """Initialize Q-values to 0 for each state-action pair."""
        self.q_values = defaultdict(lambda: np.zeros(action_space))

    def get_q_value(self, state, action):
        """Get the Q-value for a given state-action pair."""
        return self.q_values.get(state, np.zeros(len(self.q_values[0])))[action]

    def set_q_value(self, state, action, value):
        """Set the Q-value for a given state-action pair."""
        if state not in self.q_values:
            self.q_values[state] = np.zeros(len(self.q_values[0]))
        self.q_values[state][action] = value
