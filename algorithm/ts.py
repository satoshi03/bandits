import numpy as np

from .base import BaseAlgorithm


class TSAlgorithm(BaseAlgorithm):
    def __init__(self, n_arms, alpha=1.0, beta=1.0):
        self.n_arms = n_arms
        self.alpha = alpha
        self.beta = beta
        self.n = np.zeros(n_arms)
        self.m = np.zeros(n_arms)

        self.n_draws = np.zeros(n_arms)

    def select(self):
        theta = np.random.beta(a=self.n + self.alpha, b=self.m + self.beta)
        return np.argmax(theta)

    def update(self, arm_num, reward):
        if reward:
            self.n[arm_num] += 1
        else:
            self.m[arm_num] += 1
        self.n_draws[arm_num] += 1

    def __str__(self):
        return 'ThompsonSampling'
