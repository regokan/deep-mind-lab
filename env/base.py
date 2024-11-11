"""Base class for OpenAI Gym environments."""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np

from policy import BasePolicy

np.bool8 = np.bool


class BaseEnvironment(ABC):
    """Abstract class for an environment."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def reset(self) -> Tuple[Any, ...]:
        """Reset the environment to the initial state."""

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """Take an action and return the new state, reward, and done status."""

    @abstractmethod
    def get_action_space(self) -> Any:
        """Return the action space for the environment."""

    @abstractmethod
    def get_observation_space(self) -> Any:
        """Return the observation space for the environment."""

    @abstractmethod
    def render(self) -> Any:
        """Render the environment."""

    @abstractmethod
    def close(self) -> None:
        """Close the environment."""

    def watch(self, policy: BasePolicy, display: Any, steps: int = 1000) -> None:
        """Watch the agent play the environment."""
        state = self.reset()[0]
        score = 0
        img = plt.imshow(self.render())
        for _ in range(steps):
            action = policy.select_action(state)
            img.set_data(self.render())
            plt.axis("off")
            display.display(plt.gcf())
            display.clear_output(wait=True)
            state, reward, done, _, _ = self.step(action)
            score += reward
            if done:
                break
        print("Score: ", score)
