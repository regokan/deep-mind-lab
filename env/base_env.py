"""Base class for OpenAI Gym environments."""

from typing import Any, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np

from policy import BasePolicy

from .constants import _SUPPORTED_ENVIRONMENTS

np.bool8 = np.bool


class Environment:
    """Abstract class for an environment."""

    def __init__(self, name: str, render_mode: str = "rgb_array"):
        self.name = name
        if name not in _SUPPORTED_ENVIRONMENTS:
            raise ValueError(
                f"Unsupported environment: {name}, "
                f"supported environments are {', '.join(_SUPPORTED_ENVIRONMENTS.keys())}"
            )
        self.env = gym.make(_SUPPORTED_ENVIRONMENTS[name], render_mode=render_mode)

    def reset(self) -> Tuple[Any, ...]:
        """Reset the environment to the initial state."""
        return self.env.reset()

    def step(self, action: Any) -> Tuple[Any, ...]:
        """Take an action and return the new state, reward, and done status."""
        return self.env.step(action)

    def get_action_space(self) -> gym.Space:
        """Return the action space for the environment."""
        return self.env.action_space

    def get_observation_space(self) -> gym.Space:
        """Return the observation space for the environment."""
        return self.env.observation_space

    def render(self) -> Any:
        """Render the environment."""
        return self.env.render()

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

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
                print("Score: ", score)
                break
