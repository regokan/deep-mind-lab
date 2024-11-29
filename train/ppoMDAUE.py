from collections import deque
from typing import Any, Callable, Optional

import numpy as np
import progressbar as pb
import torch
from mlagents_envs.base_env import ActionSpec, ActionTuple

from env import MLAgents
from policy import BasePolicy

from .base import BaseTrainer

widget = ["training loop: ", pb.Percentage(), " ", pb.Bar(), " ", pb.ETA()]


class PPOMDAUETrainer(BaseTrainer):
    """
    Trainer for the REINFORCE policy with
        - MA: Multiple Discrete actions
        - UE: Unity Environment
    """

    def __init__(
        self,
        policy: BasePolicy,
        environment: MLAgents,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        max_steps_per_episode=1000,
        gamma=0.99,
        print_every=100,
        target_score=195.0,
        preprocess_state_fn: Callable = None,
        device="cpu",
    ):
        if not hasattr(policy, "select_action") or not hasattr(policy, "update_policy"):
            raise ValueError(
                "Policy must implement 'select_action' and 'update_policy' methods."
            )

        super().__init__(policy, environment)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_steps_per_episode = max_steps_per_episode
        self.gamma = gamma
        self.print_every = print_every
        self.target_score = target_score
        self.device = device
        self.preprocess_state_fn = preprocess_state_fn or (
            lambda x, device=None: np.array(x)
        )

    def _collect_trajectory(self, nrand=5, action_spec: ActionSpec = None):
        """
        Collect a trajectory from the environment.

        Args:
            nrand: The number of random steps to take.
            action_spec: ActionSpec defining the structure of discrete and/or continuous action spaces.
        """

        # Initialize lists for trajectory data
        state_list = []
        reward_list = []
        prob_list = []
        action_list = []

        # Reset the environment and get initial observation
        obs = self.environment.reset()
        n = obs.shape[0]  # Batch size (number of agents in parallel)
        action_size = action_spec.discrete_branches[0]

        # Start the game by taking initial steps with a placeholder action
        initial_action = ActionTuple(
            discrete=np.array([[0]] * n)
        )
        self.environment.step(initial_action)

        # Perform `nrand` random steps to explore the environment
        for _ in range(nrand):
            random_actions = ActionTuple(
                discrete=np.random.randint(
                    0,
                    action_size,
                    size=(n, 1),
                )
            )
            fr1, re1, is_done1, _, _ = self.environment.step(initial_action)
            fr2, re2, is_done2, _, _ = self.environment.step(random_actions)


        # Main loop for collecting steps until reaching max steps per episode
        for t in range(self.max_steps_per_episode):
            # Prepare input for the policy by processing the frames
            batch_input = self.preprocess_state_fn([fr1, fr2]).to(self.device)

            # Select action probabilities (π_θ) without backpropagation
            probs = (
                self.policy.select_action(batch_input).squeeze().cpu().detach().numpy()
            )

            # Sample actions for each batch based on the probability distribution
            sampled_actions = ActionTuple(
                discrete=np.array([
                    np.random.choice(action_size, 1, p=probs[i])
                    for i in range(n)
                ])
            )

            # Step the environment with selected actions
            fr1, re1, is_done, _, _ = self.environment.step(initial_action)
            fr2, re2, is_done, _, _ = self.environment.step(sampled_actions)

            # Aggregate reward
            reward = re1 + re2

            # Append data to trajectory lists
            state_list.append(batch_input)
            reward_list.append(reward)
            prob_list.append(probs)
            action_list.append(sampled_actions)

            # Stop if all agents are done (for rectangular trajectory storage)
            # @TODO is the correct way to do this?
            if is_done.all():
                break

        # Return collected trajectory data: pi_theta, states, actions, rewards, probability
        return prob_list, state_list, action_list, reward_list

    def train(self, episodes, **kwargs):
        """
        Train the policy.

        Args:
            episodes: The number of episodes to train for.
            **kwargs: Additional keyword arguments.
                - beta: The beta value for the entropy regularization.
                - action_spec:
                - future_rewards_only: Whether to use future rewards only.
                - normalize_rewards: Whether to normalize the rewards.
                - beta_decay: The beta decay value.
        """
        timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.print_every)

        beta = kwargs.get("beta", 0.01)
        action_spec = kwargs.get("action_spec", None)
        if action_spec is None:
            raise ValueError("Actions Spec must be provided.")
        
        ppo_epoch = kwargs.get("ppo_epoch", 4)
        epsilon = kwargs.get("epsilon", 0.2)
        epsilon_decay = kwargs.get("epsilon_decay", 0.995)

        for episode in range(1, episodes + 1):
            old_probs, states, actions, rewards = self._collect_trajectory(
                action_spec=action_spec,
                nrand=1
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
                    rewards, old_probs, states, beta
                )
                self.optimizer.step()

            # reduce the epsilon value to reduce exploration
            epsilon *= epsilon_decay

            if self.scheduler:
                # Check if the scheduler is of type ReduceLROnPlateau
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
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

    def evaluate(self, episodes: int, **kwargs) -> float:
        action_spec = kwargs.get("action_spec", None)

        total_rewards = 0

        for _ in range(episodes):
            _, _, _, reward_list = self._collect_trajectory(
                nrand=1, action_spec=action_spec
            )
            total_rewards += np.sum(reward_list)

        average_reward = total_rewards / episodes
        print(
            f"Evaluation: Average reward over {episodes} episodes: {average_reward:.2f}"
        )
        return average_reward
