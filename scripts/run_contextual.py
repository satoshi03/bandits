import numpy as np
import matplotlib.pyplot as plt

from arm.gauss import GaussArms
from algorithm.lin_ucb import LinUCBAlgorithm


def main():
    n = 10000
    d = 10
    K = 5

    X = np.random.random((n, d))
    theta = np.random.normal(0, 1, (K, d))

    arms = GaussArms(X, theta)
    lua = LinUCBAlgorithm(K, d)

    norms = []
    rewards = np.zeros(n)
    regrets = np.zeros(n)
    for i in range(n):
        arm_num = lua.select(X[i])
        reward = arms.draw(arm_num)
        lua.update(arm_num, X[i], reward)
        rewards[i] = reward
        regrets[i] = arms.get_max(i) - reward
        norms.append(np.linalg.norm((lua.th_hat - theta), 'fro'))

    plt.subplot(121)
    plt.plot(regrets.cumsum())
    plt.subplot(122)
    plt.plot(norms)
    plt.show()


if __name__ == "__main__":
    main()
