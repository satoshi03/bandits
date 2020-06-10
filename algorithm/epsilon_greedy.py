import numpy as np
import random

from .base import BaseAlgorithm


class EpsilonGreedyAlgorithm(BaseAlgorithm):
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.values = np.zeros(n_arms)
        self.n_draws = np.zeros(n_arms)

    def select(self):
        if self.epsilon > random.random():
            return np.argmax(self.values)
        else:
            return random.randint(1, self.n_arms) - 1

    def update(self, arm_num, reward):
        value = self.values[arm_num]
        n_draws = self.n_draws[arm_num] + 1
        value = (n_draws / (float(n_draws)  + 1) * value + (float(1) / n_draws * reward))
        self.n_draws[arm_num] = n_draws
        self.values[arm_num] = value

    def __str__(self):
        return 'EpsilonGreedy'
