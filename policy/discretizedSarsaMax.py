"""Discretized SARSA-Max policy."""

import numpy as np

from .helper import discretize
from .sarsaMax import SarsaMaxPolicy


class DiscretizedSarsaMaxPolicy(SarsaMaxPolicy):
    def __init__(
        self,
        q_table,
        action_space,
        state_grid,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay_rate=0.9995,
        min_epsilon=0.01,
    ):
        """Discretized SARSA-Max policy."""
        super().__init__(q_table, action_space, alpha, gamma, epsilon)
        self.state_grid = state_grid  # Grid for discretizing continuous states
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon

    def preprocess_state(self, state):
        """Map a continuous state to its discretized representation."""
        return tuple(discretize(state, self.state_grid))

    def select_action(self, state):
        """Epsilon-greedy action selection with discretized state."""
        state = self.preprocess_state(state)  # Discretize the state
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table.q_values[state])

    def update_policy(self, state, action, reward, next_state, done):
        """Update Q-table using SARSA-Max rule with discretized states."""
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)

        best_next_action = np.argmax(self.q_table.q_values[next_state])
        target = reward + self.gamma * self.q_table.get_q_value(
            next_state, best_next_action
        ) * (not done)
        current_q = self.q_table.get_q_value(state, action)
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table.set_q_value(state, action, new_q)

    def update_exploration(self, episode):
        """Gradually decay epsilon after each episode."""
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)
