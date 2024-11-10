from typing import Any, Tuple

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from .base import BaseEnvironment
from .constants import _SUPPORTED_UNITY_ENVIRONMENTS


class UnityAgents(BaseEnvironment):
    """Concrete implementation for Unity ML-Agents environments."""

    def __init__(
        self,
        name: str,
        worker_id: int = 0,
        seed: int = 42,
        render_mode: str = "rgb_array",
    ):
        if name not in _SUPPORTED_UNITY_ENVIRONMENTS:
            raise ValueError(
                f"Unsupported environment: {name}, "
                f"supported environments are {', '.join(_SUPPORTED_UNITY_ENVIRONMENTS.keys())}"
            )

        super().__init__(name)

        self.env = UnityEnvironment(
            file_name=_SUPPORTED_UNITY_ENVIRONMENTS[name],
            worker_id=worker_id,
            seed=seed,
            no_graphics=True,
        )
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.render_mode = render_mode

    def reset(self) -> Tuple[np.ndarray, ...]:
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        return decision_steps.obs[0]

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, dict]:
        action_tuple = ActionTuple()
        action_tuple.add_continuous(action)
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        if len(terminal_steps) > 0:
            done = True
            state = terminal_steps.obs[0]
            reward = terminal_steps.reward[0]
        else:
            done = False
            state = decision_steps.obs[0]
            reward = decision_steps.reward[0]

        return state, reward, done, {}

    def get_action_space(self) -> Any:
        return self.env.behavior_specs[self.behavior_name].action_spec

    def get_observation_space(self) -> Any:
        return self.env.behavior_specs[self.behavior_name].observation_shapes

    def render(self) -> Any:
        if self.render_mode == "rgb_array":
            decision_steps, _ = self.env.get_steps(self.behavior_name)
            return decision_steps.obs[0]
        raise NotImplementedError("Only 'rgb_array' render mode is supported")

    def close(self) -> None:
        self.env.close()
