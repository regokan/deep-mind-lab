from collections import deque

import numpy as np
import torch

from .base import BaseTrainer


class TrainerDQN(BaseTrainer):
    def __init__(
        self,
        policy,
        environment,
        max_steps_per_episode=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
    ):
        if not hasattr(policy, "select_action") or not hasattr(policy, "update_policy"):
            raise ValueError(
                "Policy must implement 'select_action' and 'update_policy' methods."
            )

        super().__init__(policy, environment)
        self.max_steps_per_episode = max_steps_per_episode
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def train(self, episodes):
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        epsilon = self.eps_start  # initialize epsilon

        for episode in range(1, episodes + 1):
            state = self.environment.reset()[0]
            score = 0
            for t in range(self.max_steps_per_episode):
                action = self.policy.select_action(state, epsilon)
                next_state, reward, done, _, _ = self.environment.step(action)
                self.policy.update_policy(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break

            scores_window.append(score)
            scores.append(score)
            epsilon = max(self.eps_end, self.eps_decay * epsilon)

            # Print average score over last 1000 episodes
            if episode % 1000 == 0:
                print(
                    f"\rEpisode {episode}\tAverage Score: {np.mean(scores_window):.2f}"
                )
            if np.mean(scores_window) >= 200.0:
                print(
                    f"\nEnvironment solved in {episode - 100} episodes!"
                    f"\tAverage Score: {np.mean(scores_window):.2f}"
                )
                torch.save(
                    self.policy.qnetwork_local.state_dict(),
                    "checkpoints/checkpoint.pth",
                )
                break

        return scores

    def evaluate(self, episodes):
        total_rewards = 0
        for _ in range(episodes):
            state = self.environment.reset()[0]
            done = False
            episode_reward = 0
            steps = 0

            while not done and steps < self.max_steps_per_episode:
                action = self.policy.select_action(
                    state, eps=0.0
                )  # Greedy action selection
                state, reward, done, _, _ = self.environment.step(action)
                episode_reward += reward
                steps += 1

            total_rewards += episode_reward

        average_reward = total_rewards / episodes
        print(
            f"Evaluation: Average reward over {episodes} episodes: {average_reward:.2f}"
        )
        return average_reward
