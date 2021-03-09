from numpy import random
import random as rand
import sys
import numpy as np

class SeqClassing:
    def __init__(self):
        # Set init input
        self.reset()

    def seed(self, num):
        print("Seed")
        print(num)
        random.seed(num)

    def reset(self):
        self.input = []
        self.curr_step = 1
        for i in range(rand.randint(3, 10)):
            # https://stackoverflow.com/a/46820635
            j = 1 if random.random() < 0.5 else -1
            self.input.append(j)
            for k in range(rand.randint(10, 20)):
                self.input.append(0)
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
