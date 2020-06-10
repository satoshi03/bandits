import numpy as np

from .base import BaseArm


class GaussArms():
    def __init__(self, X, theta):
        self.X = X
        self.theta = theta
        self.y = np.dot(self.X, self.theta.T)
        self.i = 0

    def draw(self, arm_num):
        reward = self.y[self.i, arm_num]
        self.i += 1
        return reward

    def get_max(self, i):
        return self.y[i].max()

    def reset(self):
        self.i = 0
