# Policy Module

This module provides a variety of policy classes for implementing different reinforcement learning algorithms. These policies are designed to work with environments and facilitate the learning process by selecting actions based on the current state and updating the policy based on feedback.

## Classes

### BasePolicy

- **Purpose**: Abstract base class for all policies.
- **Features**:
  - Defines the basic interface for action selection and exploration update.

### AlphaMCPolicy

- **Purpose**: Implements Monte Carlo policy with incremental Q-value updates.
- **Features**:
  - Uses epsilon-greedy action selection.
  - Updates Q-values using every-visit returns with a constant step size.

### CrossEntropyPolicy

- **Purpose**: Implements a policy using the cross-entropy method.
- **Features**:
  - Neural network-based policy.
  - Supports weight updates for exploration.

### DiscretizedSarsaMaxPolicy

- **Purpose**: Implements SARSA-Max policy with state discretization.
- **Features**:
  - Discretizes continuous states for Q-learning.
  - Uses epsilon-greedy action selection.

### DQNPolicy

- **Purpose**: Implements Deep Q-Network (DQN) policy.
- **Features**:
  - Uses neural networks to approximate Q-values.
  - Supports experience replay and target network updates.

### EveryVisitMCPolicy

- **Purpose**: Implements Monte Carlo policy using every-visit returns.
- **Features**:
  - Uses epsilon-greedy action selection.
  - Updates Q-values based on average returns.

### FirstVisitMCPolicy

- **Purpose**: Implements Monte Carlo policy using first-visit returns.
- **Features**:
  - Uses epsilon-greedy action selection.
  - Updates Q-values based on first-visit returns.

### HillClimbingPolicy

- **Purpose**: Implements a policy using hill climbing optimization.
- **Features**:
  - Stochastic policy with noise scaling for exploration.

### PPO_MDA_UE_Policy

- **Purpose**: Implements PPO policy with multiple discrete actions for Unity environments.
- **Features**:
  - Uses entropy regularization.
  - Supports policy updates with PPO algorithm.

### PPOParallelPolicy

- **Purpose**: Implements PPO policy for parallel environments.
- **Features**:
  - Uses entropy regularization.
  - Supports policy updates with PPO algorithm.

### RandomPolicy

- **Purpose**: Implements a random policy.
- **Features**:
  - Selects actions randomly, ignoring the current state.

### ReinforcePolicy

- **Purpose**: Implements the REINFORCE policy.
- **Features**:
  - Uses policy gradient methods for updates.
  - Supports action selection based on learned probabilities.

### ReinforceParallelPolicy

- **Purpose**: Implements REINFORCE policy for parallel environments.
- **Features**:
  - Uses entropy regularization.
  - Supports policy updates with REINFORCE algorithm.

### SarsaPolicy

- **Purpose**: Implements the SARSA policy.
- **Features**:
  - Uses epsilon-greedy action selection.
  - Updates Q-values using the SARSA rule.

### SarsaExpectedPolicy

- **Purpose**: Implements the Expected SARSA policy.
- **Features**:
  - Uses epsilon-greedy action selection.
  - Updates Q-values using the expected SARSA rule.

### SarsaMaxPolicy

- **Purpose**: Implements the SARSA-Max (Q-learning) policy.
- **Features**:
  - Uses epsilon-greedy action selection.
  - Updates Q-values using the SARSA-Max rule.

## Utilities

### Helper Functions

- **Purpose**: Provides utility functions for policy operations.
- **Features**:
  - Includes functions for state discretization.

## Usage

These classes can be used to implement various reinforcement learning algorithms. They provide flexibility in terms of policy configuration and learning strategies, making them suitable for a variety of tasks.

## Installation

Ensure you have the necessary dependencies installed, including `numpy` and `torch`.
