import numpy as np
from collections import defaultdict


class Agent:
    def __init__(
        self,
        nA=6,
        alpha=0.1,
        gamma=1.0,
        epsilon=1.0,
        min_epsilon=0.01,
        epsilon_decay=0.995,
    ):
        """Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - alpha: learning rate
        - gamma: discount factor
        - epsilon: initial exploration rate
        - min_epsilon: minimum exploration rate
        - epsilon_decay: decay rate for exploration
        """
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """Given the state, select an action using epsilon-greedy policy."""
        if np.random.rand() > self.epsilon:  # Exploit
            return np.argmax(self.Q[state])
        else:  # Explore
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """Update the agent's knowledge using the SARSA update rule."""
        # SARSA update rule
        next_action = self.select_action(next_state)
        td_target = reward + (self.gamma * self.Q[next_state][next_action] * (not done))
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

        # Decay epsilon after each episode if done
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
