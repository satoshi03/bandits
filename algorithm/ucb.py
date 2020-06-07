import math

from .base import BaseAlgorithm


class UCBAlgorithm(BaseAlgorithm):
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.rewards = [0.0 for i in range(n_arms)]
        self.n_draws = [0 for i in range(n_arms)]

    def select(self):
        selected = 0
        max_value = 0
        for n in range(self.n_arms):
            if self.n_draws[n] > 0:
                avg = self.rewards[n] / self.n_draws[n]
                delta = math.sqrt(2 * math.log(n+1) / self.n_draws[n])
                value = avg + delta
            else:
                value = 1e400

            if value > max_value:
                max_value = value
                selected = n
        return selected

    def update(self, arm_num, reward):
        self.n_draws[arm_num] += 1
        self.rewards[arm_num] += reward

    def __str__(self):
        return 'UCB'



