# Train Module

This module provides a set of trainer classes for implementing various reinforcement learning algorithms. These trainers are designed to work with policies and environments to facilitate the training and evaluation of reinforcement learning agents.

## Classes

### BaseTrainer

- **Purpose**: Abstract base class for all trainers.
- **Features**:
  - Defines the basic interface for training and evaluating policies.

### CrossEntropyTrainer

- **Purpose**: Implements a trainer using the cross-entropy method.
- **Features**:
  - Uses a population-based approach to optimize policy weights.
  - Selects top-performing weights to update the policy.

### TrainerHillClimbing

- **Purpose**: Implements a trainer using hill climbing optimization.
- **Features**:
  - Optimizes policy weights using stochastic hill climbing.
  - Adjusts exploration based on performance.

### PPOMDAUETrainer

- **Purpose**: Implements a PPO trainer for Unity environments with multiple discrete actions.
- **Features**:
  - Uses Proximal Policy Optimization (PPO) with entropy regularization.
  - Supports policy updates with PPO algorithm.

### PPOParallelTrainer

- **Purpose**: Implements a PPO trainer for parallel environments.
- **Features**:
  - Uses Proximal Policy Optimization (PPO) with entropy regularization.
  - Supports policy updates with PPO algorithm.

### ReinforceTrainer

- **Purpose**: Implements a trainer for the REINFORCE policy.
- **Features**:
  - Uses policy gradient methods for updates.
  - Supports action selection based on learned probabilities.

### ReinforceParallelTrainer

- **Purpose**: Implements a trainer for the REINFORCE policy with parallel environments.
- **Features**:
  - Uses entropy regularization.
  - Supports policy updates with REINFORCE algorithm.

### TrainerDQN

- **Purpose**: Implements a trainer for Deep Q-Network (DQN) policies.
- **Features**:
  - Uses experience replay and target network updates.
  - Supports epsilon-greedy action selection.

### TrainerMC

- **Purpose**: Implements a trainer for Monte Carlo methods.
- **Features**:
  - Supports first-visit and every-visit Monte Carlo updates.
  - Uses epsilon-greedy action selection.

### TrainerTD

- **Purpose**: Implements a trainer for Temporal Difference (TD) methods.
- **Features**:
  - Supports SARSA, SARSA-Max, and Expected SARSA updates.
  - Uses epsilon-greedy action selection.

## Usage

These classes can be used to train and evaluate reinforcement learning agents using various algorithms. They provide flexibility in terms of training configuration and learning strategies, making them suitable for a variety of tasks.

## Installation

Ensure you have the necessary dependencies installed, including `numpy`, `torch`, and `progressbar`.
