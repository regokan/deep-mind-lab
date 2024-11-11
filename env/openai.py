from typing import Any, Tuple

import gym

from .base import BaseEnvironment
from .constants import _SUPPORTED_OPENAI_ENVIRONMENTS


class Gym(BaseEnvironment):
    """Concrete implementation for OpenAI Gym environments."""

    def __init__(self, name: str, render_mode: str = "rgb_array"):
        if name not in _SUPPORTED_OPENAI_ENVIRONMENTS:
            raise ValueError(
                f"Unsupported environment: {name}, "
                f"supported environments are {', '.join(_SUPPORTED_OPENAI_ENVIRONMENTS.keys())}"
            )

        super().__init__(name)
        self.env = gym.make(
            _SUPPORTED_OPENAI_ENVIRONMENTS[name], render_mode=render_mode
        )

    def reset(self) -> Tuple[Any, ...]:
        return self.env.reset()

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        return self.env.step(action)

    def get_action_space(self) -> Any:
        return self.env.action_space

    def get_observation_space(self) -> Any:
        return self.env.observation_space

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
