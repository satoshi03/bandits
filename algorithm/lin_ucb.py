import numpy as np

from .base import BaseAlgorithm


class LinUCBAlgorithm(BaseAlgorithm):
    def __init__(self, K, d, alpha=0.2):
        self.K = K
        self.b = np.zeros((self.K, d))
        self.A = np.zeros((self.K, d, d))

        for a in range(self.K):
            self.A[a] = np.identity(d)

        self.th_hat = np.zeros((self.K, d))
        self.alpha = alpha

    def select(self, x_i):
        selected = 0
        max_p = 0
        for a in range(self.K):
            A_inv = np.linalg.inv(self.A[a])
            self.th_hat[a] = A_inv.dot(self.b[a])

            ta = x_i.dot(A_inv).dot(x_i)
            upperbound = self.alpha * np.sqrt(ta)

            linear_model = self.th_hat[a].dot(x_i)
            p = linear_model + upperbound
            if p > max_p:
                selected = a
                max_p = p
        return selected

    def update(self, arm_num, x_i, reward):
        self.A[arm_num] += np.outer(x_i, x_i)
        self.b[arm_num] += reward * x_i
