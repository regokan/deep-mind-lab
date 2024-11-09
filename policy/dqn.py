import random
from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from memory import ReplayBuffer

from .base import BasePolicy


class DQNPolicy(BasePolicy, ABC):
    def __init__(
        self,
        state_size,
        action_size,
        model,
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        tau=1e-3,
        lr=5e-4,
        update_every=4,
        seed=0,
        device="cpu",
    ):
        """Initialize the DQN Policy."""
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every

        # Q-Networks
        self.qnetwork_local = model(state_size, action_size, seed).to(device)
        self.qnetwork_target = model(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, device)
        self.batch_size = batch_size
        self.t_step = 0
        self.device = device

    def select_action(self, state, eps=0.0):
        """Select an action based on epsilon-greedy policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def update_policy(self, state, action, reward, next_state, done):
        """Save experience in replay memory and learn if necessary."""
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        """Update network parameters using a batch of experiences."""
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values for next states from target model
        Q_targets_next = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss and update local network
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def update_exploration(self, episode, epsilon_decay=0.995, min_epsilon=0.01):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon * epsilon_decay, min_epsilon)
