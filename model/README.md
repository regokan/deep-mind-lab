# Model Module

This module contains neural network components designed for deep learning applications using PyTorch. It includes two primary classes:

## Classes

### Conv2DBody

- **Purpose**: Implements a two-layer convolutional neural network (CNN) body.
- **Features**:
  - Configurable input, hidden, and output channels.
  - Customizable kernel size, stride, and padding for each layer.
  - Optional activation functions for hidden and output layers.
  - Weight initialization using Kaiming normal.

### FCBody

- **Purpose**: Implements a two-layer fully connected (dense) neural network body.
- **Features**:
  - Configurable input, hidden, and output sizes.
  - Optional activation functions for hidden and output layers.
  - Weight initialization using Xavier uniform.

## Usage

These classes can be imported and used to build more complex neural network architectures. They provide flexibility in terms of layer configuration and activation functions, making them suitable for a variety of tasks.
