from typing import Any, Tuple, Callable

import numpy as np
from matplotlib import pyplot as plt
from mlagents_envs.registry import default_registry
from mlagents_envs.base_env import ActionTuple
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
        return decision_steps.obs

    def step(self, action: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Take an action and 
        Returns:
            - the new state, 
            - reward, 
            - done status
            - additional info (None, matching OpenAI Gym API standard).
        After taking the action, for each agent.
        """
        # Get the number of agents currently in the environment
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        total_agents = len(decision_steps)

        # Set the actions
        self.env.set_actions(self.behavior_name, action)

        # Move the simulation forward
        self.env.step()

        # Retrieve observation, reward, and done status
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        done = np.zeros(total_agents, dtype=bool)
        reward = np.zeros(total_agents)
        state = [None] * total_agents  # Placeholder for states for all agents

        # Process done agents
        for agent_id in terminal_steps.agent_id:
            index = terminal_steps.agent_id_to_index[agent_id]
            done[agent_id] = True
            reward[agent_id] = terminal_steps.reward[index]
            state[agent_id] = terminal_steps.obs[0][index]

        # Process active agents
        for agent_id in decision_steps.agent_id:
            index = decision_steps.agent_id_to_index[agent_id]
            done[agent_id] = False
            reward[agent_id] = decision_steps.reward[index]
            state[agent_id] = decision_steps.obs[0][index]

        # Convert the list of states to a numpy array
        state = np.array(state)

        return state, reward, done, None, {}

    def get_action_space(self) -> Any:
        """Return the action space for the environment."""
        return self.env.behavior_specs[self.behavior_name].action_spec

    def get_observation_space(self) -> Any:
        """Return the observation space for the environment."""
        return self.env.behavior_specs[self.behavior_name].observation_specs

    def render(self) -> list:
        """Render the environment for visualization, returning all frames in the batch."""
        if self.render_mode == "rgb_array":
            decision_steps, _ = self.env.get_steps(self.behavior_name)
            frames = decision_steps.obs[0]  # Get all frames in the batch

            # Process each frame to ensure it’s in (height, width, channels) format
            processed_frames = []
            for frame in frames:
                if frame.shape[0] == 5:
                    frame = np.mean(frame[:3], axis=0)  # Average first 3 channels
                elif frame.ndim == 3 and frame.shape[0] in [1, 3]:
                    frame = np.transpose(frame, (1, 2, 0))  # Rearrange to (height, width, channels)
                processed_frames.append(frame)
            return processed_frames
        else:
            raise NotImplementedError("Only 'rgb_array' render mode is supported")

    def close(self) -> None:
        """Close the Unity environment."""
        self.env.close()

    def watch(self, policy, display, steps: int = 1000, preprocess_state_fn: Callable = None) -> None:
        """Watch the agent play the environment, displaying each frame separately."""
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        score = 0
        n = len(decision_steps.obs[0])

        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))  # Set up a row of subplots

        state = decision_steps.obs[0]

        for _ in range(steps):
            if preprocess_state_fn:
                state = preprocess_state_fn(state)
            action = policy.select_action(state)
            if not isinstance(action, ActionTuple):  
                action = ActionTuple(
                    discrete=action.unsqueeze(1).numpy(),
            )
            frames = self.render()  # Get all frames in the batch

            for idx, frame in enumerate(frames):
                axes[idx].imshow(frame)
                axes[idx].axis("off")  # Hide axes for a clean look

            display.display(fig)
            display.clear_output(wait=True)
            
            # Perform environment step and update score
            state, reward, done, _, _ = self.step(action)
            score += sum(reward)
            
            if done.all():
                break

        print("Average Score: ", score / n)
