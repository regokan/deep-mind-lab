# Utils Module

This module provides utility functions to support the main functionalities of the reinforcement learning framework. It includes functions for device management to ensure efficient computation.

## Functions

### get_device

- **Purpose**: Determines the best available device for computation.
- **Features**:
  - Checks for the availability of CUDA (NVIDIA GPUs), MPS (Apple Silicon GPUs), and falls back to CPU if neither is available.
  - Returns a `torch.device` object representing the best available device.

## Usage

The `get_device` function can be used to automatically select the most suitable device for running PyTorch models, ensuring optimal performance.
