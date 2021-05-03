import random
import fastrand
import sys
import numpy as np

class SeqClassing:
    def __init__(self, n = 0):
        # Set init input
        self.N = n
        self.reset()

    def seed(self, num):
        print("Seed")
        print(num)
        fastrand.pcg32_seed(num)
        random.seed(num)

    def restart(self):
        self.curr_step = 1
        return np.array([self.input[0]])

    def reset(self):
        self.input = []
        self.curr_step = 1
        # Ability to set constant number of ones
        num_of_ones = self.N
        if (self.N == 0):
            num_of_ones = fastrand.pcg32bounded(7)+3
        for i in range(num_of_ones):
            # https://stackoverflow.com/a/46820635
            self.input += [1 if random.random() < 0.5 else -1]
            self.input += [0]*(fastrand.pcg32bounded(10)+10)
        self.input += [1 if random.random() < 0.5 else -1]
        self.input_len = len(self.input)
        return np.array([self.input[0]])

    def step(self, action):
        s = sum(self.input[:self.curr_step-1])
        result_input = s >= 0
        score = 0
        done = 0
        value = 0

        if (result_input and action[0] and not action[1]):
            score += 1
        elif (not result_input and action[1] and not action[0]):
            score += 1
        else:
            score -= 1

        if self.curr_step == self.input_len:
            done = 1
        else:
            value = self.input[self.curr_step]
            self.curr_step += 1

        return np.array([value]), score/self.input_len, done, "N/A"
