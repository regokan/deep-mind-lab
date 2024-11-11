from collections import deque

import numpy as np
import torch

from .base import BaseTrainer


class CrossEntropyTrainer(BaseTrainer):
    def __init__(
        self,
        policy,
        environment,
        max_steps_per_episode=1000,
        gamma=0.99,
        pop_size=50,
        elite_frac=0.2,
        sigma=0.5,
        print_every=100,
        target_score=90.0,
        device="cpu",
    ):
        """
        PyTorch implementation of a cross-entropy method.

        Params
        ======
            policy: Policy object
            environment: Environment object
            max_steps_per_episode (int): maximum number of timesteps per episode
            gamma (float): discount rate
            print_every (int): how often to print average score (over last 100 episodes)
            pop_size (int): size of population at each iteration
            elite_frac (float): percentage of top performers to use in update
            sigma (float): standard deviation of additive noise
            device (str): device to use for training
        """
        if not hasattr(policy, "select_action") or not hasattr(
            policy, "update_exploration"
        ):
            raise ValueError(
                "Policy must implement 'select_action' and 'update_exploration' methods."
            )

        super().__init__(policy, environment)
        self.max_steps_per_episode = max_steps_per_episode
        self.gamma = gamma
        self.print_every = print_every
        self.target_score = target_score
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.sigma = sigma
        self.device = device

    def _evaluate(self, weights, gamma=1.0):
        self.policy.update_exploration(weights)
        episode_return = 0.0
        state = self.environment.reset()[0]
        for t in range(self.max_steps_per_episode):
            state = torch.from_numpy(state).float().to(self.device)
            action = self.policy.select_action(state)
            state, reward, done, _, _ = self.environment.step(action)
            episode_return += reward * gamma**t
            if done:
                break
        return episode_return

    def train(self, episodes):
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores

        n_elite = int(self.pop_size * self.elite_frac)
        best_weight = self.sigma * np.random.randn(self.policy.get_weights_dim())

        for episode in range(1, episodes + 1):
            rewards = []
            weights_pop = [
                best_weight
                + (self.sigma * np.random.randn(self.policy.get_weights_dim()))
                for _ in range(self.pop_size)
            ]
            rewards = np.array(
                [self._evaluate(weights, self.gamma) for weights in weights_pop]
            )

            elite_idxs = rewards.argsort()[-n_elite:]
            elite_weights = [weights_pop[i] for i in elite_idxs]
            best_weight = np.array(elite_weights).mean(axis=0)

            reward = self._evaluate(best_weight, gamma=1.0)
            scores_window.append(reward)
            scores.append(reward)

            if episode % self.print_every == 0:
                print(f"Episode {episode}\tAverage Score: {np.mean(scores_window):.2f}")
            if np.mean(scores_window) >= self.target_score:
                print(
                    f"Environment solved in {episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}"
                )
                self.policy.update_exploration(best_weight)
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
                action = self.policy.select_action(state)
                state, reward, done, _, _ = self.environment.step(action)
                episode_reward += reward
                steps += 1

            total_rewards += episode_reward

        average_reward = total_rewards / episodes
        print(
            f"Evaluation: Average reward over {episodes} episodes: {average_reward:.2f}"
        )
        return average_reward
