import random

from algorithm import GreedyAlgorithm, EpsilonGreedyAlgorithm, UCBAlgorithm
from arm import BernoulliArm


NUMBER_OF_ARMS = 10
NUMBER_OF_ITERS = 100


def main():
    arms = [BernoulliArm(random.random()) for i in range(0, NUMBER_OF_ARMS)]

    algo = GreedyAlgorithm(len(arms))
    rewards = 0
    for i in range(NUMBER_OF_ITERS):
        n = algo.select()
        reward = arms[n].draw()
        rewards += reward

    print("{}: {}".format(algo, rewards))

    algo = EpsilonGreedyAlgorithm(len(arms), 0.4)
    rewards = 0
    for i in range(NUMBER_OF_ITERS):
        n = algo.select()
        reward = arms[n].draw()
        algo.update(n, reward)
        rewards += reward

    print("{}: {}".format(algo, rewards))

    algo = UCBAlgorithm(len(arms))
    rewards = 0
    for i in range(NUMBER_OF_ITERS):
        n = algo.select()
        reward = arms[n].draw()
        algo.update(n, reward)
        rewards += reward

    print("{}: {}".format(algo, rewards))


if __name__ == "__main__":
    main()
