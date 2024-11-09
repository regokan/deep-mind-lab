import numpy as np

from .base import MonteCarloPolicy


class AlphaMCPolicy(MonteCarloPolicy):
    def __init__(self, q_table, action_space, gamma=0.99, epsilon=0.1, alpha=0.1):
        super().__init__(q_table, action_space)
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha  # Step size for incremental Q-value update
        self.q_table = q_table
        self.action_space = action_space

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table.q_values[state])

    def update_mc(self, episode_memory):
        """MC update using every-visit returns with a constant step size alpha."""
        G = 0
        visited_pairs = set()  # Track state-action pairs visited in this episode
        for state, action, reward in reversed(episode_memory):
            G = self.gamma * G + reward
            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                # Incremental update of Q-value with learning rate alpha
                old_q_value = self.q_table.get_q_value(state, action)
                new_q_value = old_q_value + self.alpha * (G - old_q_value)
                self.q_table.set_q_value(state, action, new_q_value)

    def update_exploration(self, episode):
        """Optionally decay epsilon over time."""
