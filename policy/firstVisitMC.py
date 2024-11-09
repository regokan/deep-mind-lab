import numpy as np

from .base import MonteCarloPolicy


# First-Visit MC Policy
class FirstVisitMCPolicy(MonteCarloPolicy):
    def __init__(self, q_table, action_space, gamma=0.99, epsilon=0.1):
        super().__init__(q_table, action_space)
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = q_table
        self.action_space = action_space
        self.returns_sum = (
            {}
        )  # Store the sum of returns for first-visit (state, action)
        self.visits_count = {}  # Store the count of first visits for (state, action)

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table.q_values[state])

    def update_mc(self, episode_memory):
        """MC update using first-visit returns."""
        G = 0
        visited_pairs = set()
        for state, action, reward in reversed(episode_memory):
            G = self.gamma * G + reward
            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                if (state, action) not in self.returns_sum:
                    self.returns_sum[(state, action)] = 0
                    self.visits_count[(state, action)] = 0
                # Incremental update only on first visit of (state, action)
                self.returns_sum[(state, action)] += G
                self.visits_count[(state, action)] += 1
                # Compute the average return and update Q-value
                avg_return = (
                    self.returns_sum[(state, action)]
                    / self.visits_count[(state, action)]
                )
                self.q_table.set_q_value(state, action, avg_return)

    def update_exploration(self, episode):
        """No exploration decay by default for MC."""
