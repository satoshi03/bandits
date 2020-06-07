import random

from .base import BaseArm


class BernoulliArm(BaseArm):
    def __init__(self, prob):
        self.prob = prob

    def draw(self):
        if self.prob > random.random():
            return 1
        else:
            return 0
