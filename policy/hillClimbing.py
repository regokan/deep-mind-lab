"""Hill Climbing policy."""

import numpy as np

from .base import BasePolicy


class HillClimbingPolicy(BasePolicy):
    def __init__(
        self, s_size, a_size, noise_scale=1e-2, min_noise_scale=1e-3, max_noise_scale=2
    ):
        self.s_size = s_size
        self.a_size = a_size
        self.noise_scale = noise_scale
        self.min_noise_scale = min_noise_scale
        self.max_noise_scale = max_noise_scale
        self.w = 1e-4 * np.random.rand(s_size, a_size)

    def _forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x) / sum(np.exp(x))

    def select_action(self, state):
        probs = self._forward(state)
        action = np.random.choice(np.arange(len(probs)), p=probs)  # stochastic policy
        return action

    def update_exploration(self, best_w: np.ndarray, explore=False):
        # Did not find better weights
        if explore:
            self.noise_scale = min(self.noise_scale * 2, self.max_noise_scale)
            self.w = best_w + self.noise_scale * np.random.randn(*self.w.shape)
        else:
            self.noise_scale = max(self.noise_scale / 2, self.min_noise_scale)
            self.w += self.noise_scale * np.random.randn(*self.w.shape)
