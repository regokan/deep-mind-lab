from agent import Agent
from monitor import interact
import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("Taxi-v3", render_mode="rgb_array")
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)
render = lambda: plt.imshow(env.render())
render()
