# Environment Module

This module provides a set of classes and utilities for interacting with various types of environments, including OpenAI Gym and Unity ML-Agents. It is designed to facilitate the development and testing of reinforcement learning algorithms.

## Classes

### BaseEnvironment

- **Purpose**: Abstract base class for defining environments.
- **Features**:
  - Defines the basic interface for environment interaction, including methods for reset, step, rendering, and closing.

### Gym

- **Purpose**: Concrete implementation for OpenAI Gym environments.
- **Features**:
  - Supports a variety of classic control and Atari environments.
  - Provides methods for environment interaction and rendering.

### MultiFrameGym

- **Purpose**: Extends Gym to handle multiple frames for input.
- **Features**:
  - Allows for preprocessing and handling of multiple frames as input to the policy.

### ParallelGym

- **Purpose**: Parallelized version of the Gym environment.
- **Features**:
  - Enables running multiple instances of a Gym environment in parallel for faster data collection.

### MLAgents

- **Purpose**: Environment using Unity's default registry.
- **Features**:
  - Supports Unity environments registered in the default registry.
  - Provides methods for environment interaction and rendering.

### UnityAgents

- **Purpose**: Concrete implementation for Unity ML-Agents environments.
- **Features**:
  - Supports Unity environments with continuous and discrete action spaces.
  - Provides methods for environment interaction and rendering.

## Utilities

### CloudpickleWrapper

- **Purpose**: Wrapper to enable multiprocessing with cloudpickle serialization.
- **Features**:
  - Facilitates the serialization of environment instances for parallel processing.

### Worker

- **Purpose**: Worker process to handle environment interactions asynchronously.
- **Features**:
  - Manages environment steps and resets in a separate process for parallel execution.

## Constants

- **Supported Environments**: Lists the supported OpenAI Gym and Unity environments.

## Usage

These classes can be used to create and manage environments for training reinforcement learning agents. They provide flexibility in terms of environment configuration and interaction, making them suitable for a variety of tasks.

## Installation

Ensure you have the necessary dependencies installed, including `gym`, `mlagents_envs`, and `matplotlib`.
