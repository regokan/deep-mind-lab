from collections import deque
from typing import Callable

import numpy as np
import progressbar as pb

from .base import BaseTrainer

widget = ["training loop: ", pb.Percentage(), " ", pb.Bar(), " ", pb.ETA()]


class ReinforceParallelTrainer(BaseTrainer):
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

    def _collect_trajectory(self, nrand=5, action1=None, action2=None):

        # number of parallel instances
        n = len(self.environment.processes)

        # initialize returning lists and start the game!
        state_list = []
        reward_list = []
        prob_list = []
        action_list = []

        self.environment.reset()

        # start all parallel agents
        self.environment.step([1] * n)

        # perform nrand random steps
        for _ in range(nrand):
            fr1, re1, _, _ = self.environment.step(
                np.random.choice([action1, action2], n)
            )
            fr2, re2, _, _ = self.environment.step([0] * n)

        for t in range(self.max_steps_per_episode):

            # prepare the input
            # preprocess_batch properly converts two frames into
            # the proper input for the model
            batch_input = self.preprocess_state_fn([fr1, fr2]).to(self.device)

            # probs will only be used as the pi_old
            # no gradient propagation is needed
            # so we move it to the cpu
            probs = (
                self.policy.select_action(batch_input).squeeze().cpu().detach().numpy()
            )

            action = np.where(np.random.rand(n) < probs, action1, action2)
            probs = np.where(action == action1, probs, 1.0 - probs)

            # advance the game (0=no action)
            # we take one action and skip game forward
            fr1, re1, is_done, _ = self.environment.step(action)
            fr2, re2, is_done, _ = self.environment.step([0] * n)

            reward = re1 + re2

            # store the result
            state_list.append(batch_input)
            reward_list.append(reward)
            prob_list.append(probs)
            action_list.append(action)

            # stop if any of the trajectories is done
            # we want all the lists to be retangular
            if is_done.any():
                break

        # return pi_theta, states, actions, rewards, probability
        return prob_list, state_list, action_list, reward_list

    def train(self, episodes, **kwargs):
        timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.print_every)

        beta = kwargs.get("beta", 0.01)
        action1 = kwargs.get("action1", None)
        action2 = kwargs.get("action2", None)

        for episode in range(1, episodes + 1):
            old_probs, states, actions, rewards = self._collect_trajectory(
                action1=action1,
                action2=action2,
            )

            total_rewards = np.sum(rewards, axis=0)

            scores_window.append(np.mean(total_rewards))
            scores.append(np.mean(total_rewards))

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
                rewards, old_probs, states, actions, beta, action1, action2
            )
            self.optimizer.step()

            # the regulation term also reduces
            # this reduces exploration in later runs
            beta *= kwargs.get("beta_decay", 0.995)

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

    def evaluate(self, episodes: int, **kwargs) -> float:
        action1 = kwargs.get("action1", None)
        action2 = kwargs.get("action2", None)

        total_rewards = 0
        episodes_per_env = episodes // self.environment.n_envs
        extra_episodes = episodes % self.environment.n_envs

        for _ in range(episodes_per_env):
            # Set nrand=0 for evaluation to prevent random initial steps
            _, _, _, reward_list = self._collect_trajectory(
                nrand=1, action1=action1, action2=action2
            )
            episode_rewards = np.sum(
                reward_list, axis=0
            )  # Summing over time steps for each environment
            total_rewards += np.sum(
                episode_rewards
            )  # Sum across all parallel environments

        # Handle any remaining episodes if episodes isn't a multiple of n_envs
        if extra_episodes > 0:
            for _ in range(extra_episodes):
                _, _, _, reward_list = self._collect_trajectory(
                    nrand=1, action1=action1, action2=action2
                )
                episode_reward = np.sum(reward_list)
                total_rewards += episode_reward

        average_reward = total_rewards / episodes
        print(
            f"Evaluation: Average reward over {episodes} episodes: {average_reward:.2f}"
        )
        return average_reward
