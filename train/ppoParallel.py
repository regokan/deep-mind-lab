from collections import deque
from typing import Any, Callable, Optional

import numpy as np
import progressbar as pb
import torch

from env import ParallelGym
from policy import BasePolicy

from .reinforceParallel import ReinforceParallelTrainer

widget = ["training loop: ", pb.Percentage(), " ", pb.Bar(), " ", pb.ETA()]


class PPOParallelTrainer(ReinforceParallelTrainer):
    """
    Trainer for the PPO policy with parallel environments.
    """

    def __init__(
        self,
        policy: BasePolicy,
        environment: ParallelGym,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        max_steps_per_episode=1000,
        gamma=0.99,
        print_every=100,
        target_score=195.0,
        preprocess_state_fn: Callable = None,
        device="cpu",
    ):
        super().__init__(
            policy,
            environment,
            optimizer,
            scheduler,
            max_steps_per_episode,
            gamma,
            print_every,
            target_score,
            preprocess_state_fn,
            device,
        )

    def train(self, episodes, **kwargs):
        """
        Train the policy.

        Args:
            episodes: The number of episodes to train for.
            **kwargs: Additional keyword arguments.
                - beta: The beta value for the entropy regularization.
                - action1: The first action to take.
                - action2: The second action to take.
                - future_rewards_only: Whether to use future rewards only.
                - normalize_rewards: Whether to normalize the rewards.
                - beta_decay: The beta decay value.
                - ppo_epoch: The number of PPO epochs to train for.
                - epsilon: The epsilon value for the PPO update.
                - epsilon_decay: The epsilon decay value.
        """
        timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.print_every)

        beta = kwargs.get("beta", 0.01)
        action1 = kwargs.get("action1", None)
        action2 = kwargs.get("action2", None)

        if action1 is None or action2 is None:
            raise ValueError("Actions must be provided.")

        ppo_epoch = kwargs.get("ppo_epoch", 4)
        epsilon = kwargs.get("epsilon", 0.2)
        epsilon_decay = kwargs.get("epsilon_decay", 0.995)

        for episode in range(1, episodes + 1):
            old_probs, states, actions, rewards = self._collect_trajectory(
                action1=action1,
                action2=action2,
            )

            total_rewards = np.sum(rewards, axis=0)

            scores_window.append(np.mean(total_rewards))
            scores.append(np.mean(total_rewards))

            for _ in range(ppo_epoch):
                discount = self.gamma ** np.arange(len(rewards))
                rewards = np.asarray(rewards) * discount[:, np.newaxis]

                if kwargs.get("future_rewards_only", False):
                    # convert rewards to future rewards
                    rewards = rewards[::-1].cumsum(axis=0)[::-1]

                if kwargs.get("normalize_rewards", False):
                    mean = np.mean(rewards, axis=1)
                    std = np.std(rewards, axis=1) + 1.0e-10

                    rewards = (rewards - mean[:, np.newaxis]) / std[:, np.newaxis]

                self.optimizer.zero_grad()
                self.policy.update_policy(
                    rewards, old_probs, states, actions, beta, action1, action2, epsilon
                )
                self.optimizer.step()

            # the regulation term also reduces
            # this reduces exploration in later runs
            beta *= kwargs.get("beta_decay", 0.995)

            # reduce the epsilon value to reduce exploration
            epsilon *= epsilon_decay

            if self.scheduler:
                # Check if the scheduler is of type ReduceLROnPlateau
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(np.mean(scores_window))
                else:
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
