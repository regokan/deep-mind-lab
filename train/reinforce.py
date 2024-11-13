from collections import deque

import numpy as np
import progressbar as pb

from .base import BaseTrainer

widget = ["training loop: ", pb.Percentage(), " ", pb.Bar(), " ", pb.ETA()]


class ReinforceTrainer(BaseTrainer):
    def __init__(
        self,
        policy,
        environment,
        optimizer,
        scheduler=None,
        max_steps_per_episode=1000,
        gamma=0.99,
        print_every=100,
        target_score=195.0,
    ):
        if not hasattr(policy, "select_action") or not hasattr(
            policy, "update_exploration"
        ):
            raise ValueError(
                "Policy must implement 'select_action' and 'update_exploration' methods."
            )

        super().__init__(policy, environment)
        self.optimizer = optimizer
        self.max_steps_per_episode = max_steps_per_episode
        self.gamma = gamma
        self.print_every = print_every
        self.target_score = target_score
        self.scheduler = scheduler

    def train(self, episodes, **kwargs):
        timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores

        for episode in range(1, episodes + 1):
            rewards = []
            state = self.environment.reset()[0]
            for _ in range(self.max_steps_per_episode):
                action = self.policy.select_action(state)
                next_state, reward, done, _, _ = self.environment.step(action)
                state = next_state
                rewards.append(reward)
                if done:
                    break

            scores_window.append(sum(rewards))
            scores.append(sum(rewards))

            if kwargs.get("future_rewards_only", False):
                # convert rewards to future rewards
                rewards = np.array(rewards)
                rewards = rewards[::-1].cumsum(axis=0)[::-1]

            if kwargs.get("normalize_rewards", False):
                mean = np.mean(rewards)
                std = np.std(rewards) + 1.0e-10

                rewards = (rewards - mean) / std

            discounts = [self.gamma**i for i in range(len(rewards) + 1)]
            R = sum(a * b for (a, b) in zip(discounts, rewards))

            self.optimizer.zero_grad()
            self.policy.update_exploration(R)
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            if episode % self.print_every == 0:
                if self.scheduler:
                    current_lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"Episode {episode}\tAverage Score: {np.mean(scores_window):.2f}\tLearning Rate: {current_lr:.6f}"
                    )
                else:
                    print(
                        f"Episode {episode}\tAverage Score: {np.mean(scores_window):.2f}"
                    )
            if np.mean(scores_window) >= self.target_score:
                print(
                    f"Environment solved in {episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}"
                )
                break

            # update progress widget bar
            timer.update(episode)

        timer.finish()
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
