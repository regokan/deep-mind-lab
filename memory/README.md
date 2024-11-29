# Memory Module

This module provides components for managing experience replay in reinforcement learning applications. It includes the following class:

## Classes

### ReplayBuffer

- **Purpose**: Implements a fixed-size buffer to store experience tuples for training reinforcement learning models.
- **Features**:
  - Stores experiences as tuples of state, action, reward, next state, and done flag.
  - Allows for random sampling of experience batches for training.
  - Configurable buffer size and batch size.
  - Supports operation on specified device (CPU or GPU).

## Usage

The `ReplayBuffer` class can be used to store and sample experiences during the training of reinforcement learning agents. It helps in breaking the correlation between consecutive experiences by providing random samples for training.
