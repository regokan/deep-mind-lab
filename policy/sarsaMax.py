"""SARSA-Max policy."""

import numpy as np

from .base import ValueBasedPolicy


class SarsaMaxPolicy(ValueBasedPolicy):
    def __init__(self, q_table, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        """SARSA-Max policy."""
        super().__init__(q_table, action_space)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table.q_values[state])

    def update_policy(self, state, action, reward, next_state, done):
        """Update Q-table using SARSA-Max (Q-learning) rule."""
        best_next_action = np.argmax(self.q_table.q_values[next_state])
        target = reward + self.gamma * self.q_table.get_q_value(
            next_state, best_next_action
        ) * (not done)
        current_q = self.q_table.get_q_value(state, action)
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table.set_q_value(state, action, new_q)

    def update_exploration(self, episode):
        """No exploration decay by default for SARSA-Max."""
