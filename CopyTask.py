from numpy import random
import random as rand
import sys
import numpy as np

class CopyTask:
    def __init__(self, elements = 3, sequences = 5):
        self.elements = elements
        self.sequences = sequences
        self.current_step = 0
        self.resetInput()
        self.memory = []
        self.stdio = []

    def resetInput(self):
        # Randomize sizes using given numbers as max
        elements = rand.randint(3, self.elements)
        sequences = rand.randint(3, self.sequences)

        # Handle Start/Delimiter
        self.rows = elements+2
        self.columns = sequences+2
        self.lastStep = self.columns+sequences-1

        # Generate formatted random sequence
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
        self.output = self.output[::-1]

    def seed(self, num):
        print("Seed")
        print(num)
        random.seed(num)

    def reset(self):
        self.resetInput()
        self.current_step = 0
        self.stdio = []
        self.memory = []
        return self.input

    def step(self, action):
        success = False
        score = 0
        # There are two different steps
        # Inputting steps
        # Outputting steps
        # Give data for given column of step
        if (self.current_step < self.columns):
            input = self.input[self.current_step]
            # Only increase score if push happens in valid area
            # this excludes the columns where the delimiters occur
            if (action and self.current_step != 0 and self.current_step != self.columns-1):
                score += 1
            elif (not action and (self.current_step == 0 or self.current_step == self.columns-1)):
                score += 1
            else:
                score -= 1
        else:
            input = np.array([0] * self.rows)
            mod_step = (self.current_step-2)%len(self.output)
            # Preform pop if action is 0
            if (not action and self.memory):
                pop = self.memory.pop()
                self.stdio.append(pop)
                score += 1
            else:
                score -= 1
            # Check for final state
            if (np.all(self.stdio == self.output)):
                success = True
                # print(self.stdio, self.output)
                # Optionally quit
                # quit()
                score += 1

        # Push to memory if action is set
        if (action):
            self.memory.append(input[2:])

        # Set to done on last step
        done = False
        if (self.current_step >= self.lastStep):
            done = True

        self.current_step += 1

        return input, score, done, "N/A", success
