"""Trainer classes for Temporal Difference methods."""

from policy import SarsaExpectedPolicy, SarsaMaxPolicy, SarsaPolicy

from .base import BaseTrainer


class TrainerTD(BaseTrainer):
    def __init__(self, policy, environment, max_steps_per_episode=100):
        if not isinstance(policy, (SarsaPolicy, SarsaMaxPolicy, SarsaExpectedPolicy)):
            raise ValueError(
                "Policy must be a Temporal Difference policy for TrainerTD."
            )
        super().__init__(policy, environment)
        self.max_steps_per_episode = max_steps_per_episode

    def train(self, episodes):
        for episode in range(episodes):
            state = self.environment.reset()[0]
            done = False
            steps = 0

            while not done and steps < self.max_steps_per_episode:
                action = self.policy.select_action(state)
                next_state, reward, done, _, _ = self.environment.step(action)

                # Immediate policy update for TD methods
                self.policy.update_policy(state, action, reward, next_state, done)

                state = next_state
                steps += 1

            # Update exploration for policies with decay
            self.policy.update_exploration(episode)

    def evaluate(self, episodes):
        total_rewards = 0
        for _ in range(episodes):
            state = self.environment.reset()[0]
            done = False
            episode_reward = 0
            steps = 0

            while not done and steps < self.max_steps_per_episode:
                action = self.policy.select_action(state)
                state, reward, done, _, _ = self.environment.step(action)
                episode_reward += reward
                steps += 1

            total_rewards += episode_reward

        average_reward = total_rewards / episodes
        print(f"Evaluation: Average reward over {episodes} episodes: {average_reward}")
