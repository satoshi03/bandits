import random

from .base import BaseAlgorithm


class GreedyAlgorithm(BaseAlgorithm):
    def __init__(self, n_arms):
        self.n_arms = n_arms

    def select(self):
        return random.randint(1, self.n_arms) - 1

    def update(self):
        return

    def __str__(self):
        return 'Greedy'
