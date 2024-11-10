from typing import Any, Tuple

import numpy as np
from matplotlib import pyplot as plt
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.registry import default_registry

from policy import BasePolicy

from .base import BaseEnvironment


class MLAgents(BaseEnvironment):
    """Environment using Unity's default registry."""

    def __init__(self, env_id: str, render_mode: str = "rgb_array"):
        super().__init__(env_id)
        if env_id not in default_registry:
            raise ValueError(
                f"Unsupported environment ID: {env_id}. Available environments are {', '.join(default_registry.keys())}."
            )
        self.env = default_registry[env_id].make()
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.render_mode = render_mode

        decision_steps, _ = self.env.get_steps(self.behavior_name)
        self.num_agents = decision_steps.obs[0].shape[0]
        self.num_frames = decision_steps.obs[0].shape[1]

    def reset(self) -> Tuple[np.ndarray, ...]:
        """Reset the Unity environment to the initial state."""
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        # Stacking the observations of all agents
        return decision_steps.obs[0]

    def step(self, action: dict) -> Tuple[np.ndarray, float, bool, dict]:
        """Take an action and return the new state, reward, and done status."""
        # Get the number of agents currently in the environment
        decision_steps, _ = self.env.get_steps(self.behavior_name)

        # Set the actions
        self.env.set_actions(self.behavior_name, action)

        # Move the simulation forward
        self.env.step()

        # Retrieve observation, reward, and done status
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        info = None
        if len(terminal_steps) > 0:
            done = True
            state = terminal_steps.obs[0]
            reward = terminal_steps.reward[0]
        else:
            done = False
            state = decision_steps.obs[0]
            reward = decision_steps.reward[0]

        return state, reward, done, info, {}

    def get_action_space(self) -> Any:
        """Return the action space for the environment."""
        return self.env.behavior_specs[self.behavior_name].action_spec

    def get_observation_space(self) -> Any:
        """Return the observation space for the environment."""
        return self.env.behavior_specs[self.behavior_name].observation_specs

    def render(self) -> Any:
        """Render the environment for visualization."""
        if self.render_mode == "rgb_array":
            decision_steps, _ = self.env.get_steps(self.behavior_name)
            frame = decision_steps.obs[
                0
            ]  # Assuming first observation is an image-like observation

            # Select the first observation in the batch
            frame = frame[0]  # Get the first item in the batch

            # If there are 5 channels, we can either select the first 3 or take an average
            if frame.shape[0] == 5:
                frame = np.mean(
                    frame[:3], axis=0
                )  # Average first 3 channels to form an RGB-like grayscale image

            # Reshape frame to (height, width, channels) if it’s in (channels, height, width) format
            elif frame.ndim == 3 and frame.shape[0] in [1, 3]:
                frame = np.transpose(
                    frame, (1, 2, 0)
                )  # Rearrange to (height, width, channels)

            return frame
        else:
            raise NotImplementedError("Only 'rgb_array' render mode is supported")

    def close(self) -> None:
        """Close the Unity environment."""
        self.env.close()

    def watch(self, policy: BasePolicy, display: Any, steps: int = 1000) -> None:
        """Watch the agent play the environment."""
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        score = 0
        img = plt.imshow(self.render())
        for _ in range(steps):
            action = policy.select_action(len(decision_steps))
            img.set_data(self.render())
            plt.axis("off")
            display.display(plt.gcf())
            display.clear_output(wait=True)
            _, reward, done, _, _ = self.step(action)
            score += reward
            if done:
                print("Score: ", score)
                break
