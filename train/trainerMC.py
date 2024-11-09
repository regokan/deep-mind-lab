"""Trainer classes for Monte Carlo methods."""

from policy import AlphaMCPolicy, EveryVisitMCPolicy, FirstVisitMCPolicy

from .base import BaseTrainer


class TrainerMC(BaseTrainer):
    def __init__(self, policy, environment, max_steps_per_episode=100, verbose=False):
        if not isinstance(
            policy, (FirstVisitMCPolicy, EveryVisitMCPolicy, AlphaMCPolicy)
        ):
            raise ValueError("Policy must be a Monte Carlo policy for TrainerMC.")
        super().__init__(policy, environment)
        self.max_steps_per_episode = max_steps_per_episode
        self.verbose = verbose

    def train(self, episodes):
        for episode in range(episodes):
            state = self.environment.reset()[0]
            episode_memory = []
            done = False
            steps = 0

            while not done and steps < self.max_steps_per_episode:
                action = self.policy.select_action(state)
                next_state, reward, done, _, _ = self.environment.step(action)
                episode_memory.append((state, action, reward))
                state = next_state
                steps += 1

            # After each episode, update policy based on the episode memory
            self.policy.update_mc(episode_memory)
            self.policy.update_exploration(episode)

            if self.verbose:
                print(f"Episode {episode + 1}/{episodes} completed with {steps} steps.")

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
        return average_reward  # Return average reward for further analysis
