import random
import fastrand
import sys
import numpy as np

# Input:
# Input String, Count
# Output
# No Action, Push, Pop, Dequeue

class SeqRecall:
    def __init__(self, count = 10, seq = 5):
        # Set init input
        self.count_n = count
        self.seq_n = seq
        self.reset()

    def seed(self, num):
        print("Seed")
        print(num)
        fastrand.pcg32_seed(num)
        random.seed(num)

    def restart(self):
        self.count = self.initCount
        self.curr_step = 1
        return np.array([self.input[0], 0])

    def reset(self):
        self.initCount = self.count_n;
        #self.initCount = fastrand.pcg32bounded(self.count_n)+1
        self.count = self.initCount
        seq_len = self.seq_n
        #seq_len = fastrand.pcg32bounded(self.seq_n)+1
        self.input = []
        for i in range(seq_len):
            # https://stackoverflow.com/a/46820635
            self.input += [1 if random.random() < 0.5 else -1]
        self.input_len = len(self.input)
        self.num_steps = self.seq_n * 2 + self.count
        self.curr_step = 1
        return np.array([self.input[0], 0])

    def decode_action(self, action):
        if (np.zeros(action)): # Do nothing
            return 0 # Do nothing
        elif (action[0] and (not (action[1] or action[2]))):
            return 1 # Pop
        elif (action[1] and (not (action[0] or action[2]))):
            return 2 # Push
        elif (action[2] and (not (action[0] or action[1]))):
            return 3 # Dequeue
        else:
            return 4 # Invalid Setting

    def step(self, action):
        # Push action[0]
        # Pop action[1]
        # Dequeue action[2]
        score = 0
        done = 0
        value = 0
        count = 0
        selected_action = self.decode_action(action)

        # Phases
        if (self.curr_step < self.input_len and selected_action == 1):
            score += 1 # Input
        elif (self.curr_step < (self.input_len + self.count) and selected_action == 0):
            score += 1 # Hold while count down
            count = self.count
            self.count -= 1
        elif (self.curr_step > (self.input_len + self.count) and selected_action == 3):
            score += 1 # Dequeue 
        else:
            score -= 1 # Invalid Setting

        if self.curr_step == self.num_steps:
            done = 1 # End
        elif self.curr_step < self.input_len:
            value = self.input[self.curr_step] # Give Input
            self.curr_step += 1
        else:
            self.curr_step += 1

        return np.array([value, count]), score/self.num_steps, done, "N/A"
