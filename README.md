# Deep Mind Lab

This project is a comprehensive deep reinforcement learning framework designed to facilitate the development, training, and evaluation of deep reinforcement learning agents. It includes various modules for environments, policies, training, and utilities, as well as examples demonstrating the application of different algorithms.

## Modules

### [Model Module](model/README.md)

- Contains neural network components designed for deep learning applications using PyTorch.

### [Environment Module](env/README.md)

- Provides classes and utilities for interacting with various types of environments, including OpenAI Gym and Unity ML-Agents.

### [Memory Module](memory/README.md)

- Manages experience replay in reinforcement learning applications.

### [Policy Module](policy/README.md)

- Offers a variety of policy classes for implementing different reinforcement learning algorithms.

### [QTable Module](qtable/README.md)

- Manages Q-tables for value-based reinforcement learning algorithms.

### [Train Module](train/README.md)

- Provides trainer classes for implementing various reinforcement learning algorithms.

### [Utils Module](utils/README.md)

- Includes utility functions to support the main functionalities of the framework, such as device management.

## Examples

The `examples` folder contains Jupyter notebooks demonstrating the application of different reinforcement learning algorithms using this framework. Each example is designed to showcase a specific algorithm and environment.

### Blackjack

- **Monte Carlo**: This notebook demonstrates the implementation of various Monte Carlo methods in the Blackjack environment. It explores the use of first-visit and every-visit Monte Carlo techniques to estimate the optimal policy and state-value functions.

### CartPole

- **Hill Climbing**: Demonstrates the Hill Climbing optimization method on the CartPole environment, showcasing how simple optimization techniques can be applied to reinforcement learning.
- **Reinforce**: Implements the REINFORCE policy gradient method on the CartPole environment, illustrating the use of policy gradients for training agents.

### CliffWalking

- **Temporal Difference**: This notebook explores various Temporal Difference (TD) methods in the CliffWalking environment. It covers SARSA, SARSA-Max, and Expected SARSA, providing insights into TD learning.

### GridWorld

- **Reinforce**: Demonstrates the application of the REINFORCE policy gradient method in a GridWorld environment, highlighting the use of policy gradients in grid-based tasks.

### LunarLander

- **Deep Q Network (DQN)**: Implements a DQN agent with OpenAI Gym's LunarLander-v2 environment, showcasing the use of deep learning for Q-value approximation and experience replay.

### MountainCar

- **Cross Entropy**: Demonstrates the Cross Entropy method on the MountainCarContinuous environment, illustrating a population-based optimization approach.
- **Discretization**: Explores the discretization of continuous state and action spaces in the MountainCar environment, enabling the application of discrete-space algorithms.

### Pong

- **Reinforce**: Implements the REINFORCE policy gradient method on the Pong environment, demonstrating the application of policy gradients in a classic Atari game.
- **Proximal Policy Optimization (PPO)**: Implements the PPO algorithm on the Pong environment, showcasing advanced policy optimization techniques and the use of entropy regularization.

### Taxi

- **Temporal Difference**: This notebook applies various Temporal Difference methods in the Taxi environment, demonstrating the use of TD learning in a grid-based navigation task.

## Future Development

We are planning to expand the framework with the following advanced reinforcement learning algorithms:

1. **(Double/Dueling/Prioritized) Deep Q-Learning (DQN)**

   - Enhancements to DQN improving stability (Double), separating value and advantage functions (Dueling), and prioritizing replay samples (Prioritized).

2. **Categorical DQN (C51)**

   - Extends DQN to predict a distribution over returns (not just the mean), using a categorical representation.

3. **Quantile Regression DQN (QR-DQN)**

   - Models the distribution of returns using quantiles, improving exploration and robustness.

4. **(Continuous/Discrete) Synchronous Advantage Actor Critic (A2C)**

   - Actor-Critic algorithm that synchronously collects and uses data, balancing exploration (actor) and value estimation (critic).

5. **Synchronous N-Step Q-Learning (N-Step DQN)**

   - Extends Q-learning to use multi-step returns for better credit assignment over time.

6. **Deep Deterministic Policy Gradient (DDPG)**

   - Actor-Critic method for continuous action spaces using deterministic policies and off-policy learning.

7. **The Option-Critic Architecture (OC)**

   - Learns high-level options (sub-policies) and their termination conditions for hierarchical reinforcement learning.

8. **Twined Delayed DDPG (TD3)**
   - Improves DDPG by adding target smoothing, delayed policy updates, and twin critics to reduce overestimation bias.

These additions will enhance the framework's capabilities, allowing for more sophisticated and varied reinforcement learning experiments in environments like Unity ML Agents and OpenAI Gym.

## Installation

To use this framework, ensure you have the necessary dependencies installed. This project uses Poetry for dependency management. You can set up the environment by running:

This will create a virtual environment and install all the required dependencies specified in the `pyproject.toml` file.

## Usage

Each module and example is designed to be modular and easy to integrate into your own projects. Refer to the module-specific READMEs for detailed usage instructions.

## Contributing

If you encounter any issues or have suggestions for improvements, feel free to [create a new issue](https://github.com/regokan/deep-mind-lab/issues/new) or reach out to the contributors of the project. We welcome contributions and feedback to help improve the framework.

## License

This project is open-source and available under the MIT License.
