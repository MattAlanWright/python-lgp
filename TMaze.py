from numpy import random
import random as rand
import sys
import numpy as np

class TMaze:
    def __init__(self, n = 30):
        # Set init input
        self.N = n
        self.reset()

    def seed(self, num):
        print("Seed")
        print(num)
        random.seed(num)

    def reset(self):
        # https://stackoverflow.com/a/46820635
        self.action = 1 if random.random() < 0.5 else -1
        self.count = rand.randint(1, self.N)
        return np.array([self.action, self.count])

    def compareAction(self, action):
        if (self.action == 1):
            return action[0]
        else:
            return action[1]

    def step(self, action):
        result = self.compareAction(action)
        score = 0
        done = 0

        if (self.count <= 0):
            done = 1
            score = -1
            if (result):
                score = 1
        elif (result):
            score = -1
        else:
            score = 1
            
        self.count -= 1
        return np.array([0, self.count]), score/(self.N+1), done, "N/A"
