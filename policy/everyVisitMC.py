import numpy as np

from .base import MonteCarloPolicy


class EveryVisitMCPolicy(MonteCarloPolicy):
    def __init__(self, q_table, action_space, gamma=0.99, epsilon=0.1):
        super().__init__(q_table, action_space)
        self.gamma = gamma
        self.epsilon = epsilon
        self.returns_sum = {}  # Store the sum of returns for every (state, action)
        self.visits_count = {}  # Store the count of visits for every (state, action)

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table.q_values[state])

    def update_mc(self, episode_memory):
        """MC update using every-visit returns."""
        G = 0
        for state, action, reward in reversed(episode_memory):
            G = self.gamma * G + reward
            if (state, action) not in self.returns_sum:
                self.returns_sum[(state, action)] = 0
                self.visits_count[(state, action)] = 0
            # Incremental update of returns_sum and visit counts
            self.returns_sum[(state, action)] += G
            self.visits_count[(state, action)] += 1
            # Compute the average return and update Q-value
            avg_return = (
                self.returns_sum[(state, action)] / self.visits_count[(state, action)]
            )
            self.q_table.set_q_value(state, action, avg_return)

    def update_exploration(self, episode):
        """No exploration decay by default for MC."""
