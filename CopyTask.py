from numpy import random
import numpy as np

class CopyTask:
    def __init__(self, elements = 3, sequences = 5):
        self.current_step = 0
        # Handle Start/Delimiter
        self.rows = elements+2
        self.columns = sequences+2
        self.resetInput()
        self.lastStep = self.columns+sequences

    def resetInput(self):
        rows = self.rows
        columns = self.columns
        self.input = random.randint(2, size=(rows, columns))
        self.input[0] = [0] * columns
        self.input[1] = [0] * columns
        self.input[0][0] = 1
        self.input[1][columns-1] = 1
        self.input = self.input.transpose()

    def seed(self, num):
        print("Seed")
        print(num)
        random.seed(num)

    def reset(self):
        self.resetInput()
        self.current_step = 0
        return self.input

    def step(self, action):
        # Each step is a column of data
        # Last half of steps is the resulting data

        # Give data for given column
        if (self.current_step < self.columns):
            input = self.input[self.current_step]

        score = 0

        # If input is complete run through output again
        if (self.current_step >= self.columns):
            for index,element in enumerate(self.input[self.current_step%self.columns]):
                # Offset based on current step
                if (element == action[index]):
                    score += 1

        # Set to done on last step
        done = False
        if (self.current_step >= self.lastStep):
            done = True
        self.current_step += 1

        return self.input, score, done, "N/A"
