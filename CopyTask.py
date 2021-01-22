from numpy import random
import sys
import numpy as np

class CopyTask:
    def __init__(self, elements = 3, sequences = 5):
        self.current_step = 0
        # Handle Start/Delimiter
        self.rows = elements+2
        self.columns = sequences+2
        self.resetInput()
        self.lastStep = self.columns+sequences-1
        self.memory = []

    def resetInput(self):
        rows = self.rows
        columns = self.columns
        self.input = random.randint(2, size=(rows, columns))
        self.input[0] = [0] * columns
        self.input[1] = [0] * columns
        self.input = self.input.transpose()
        self.input[0] = [0] * rows
        self.input[columns-1] = [0] * rows
        self.input[0][0] = 1
        self.input[columns-1][1] = 1
        # Slice off padding for comparing
        self.output = self.input[1:len(self.input)-1,2:]

    def seed(self, num):
        print("Seed")
        print(num)
        random.seed(num)

    def reset(self):
        self.resetInput()
        self.current_step = 0
        return self.input

    def step(self, action):
        score = 0
        # There are two different steps
        # Inputting steps
        # Outputting steps
        # Give data for given column of step
        if (self.current_step < self.columns):
            input = self.input[self.current_step]
        else:
            input = np.array([0] * self.rows)
            mod_step = (self.current_step-2)%len(self.output)
            if (len(action) < len(self.output[mod_step])):
                print("ERROR: Register space smaller then element size")
                sys.exit()
            else:
                score += np.sum(self.output[mod_step] == action[:len(self.output[mod_step])])

        # Set to done on last step
        done = False
        if (self.current_step >= self.lastStep):
            done = True

        self.current_step += 1

        return input, score, done, "N/A"
