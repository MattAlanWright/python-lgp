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
        
        self.numSteps = self.columns+sequences-1

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
        return self.input[0]

    def step(self, action):
        score = 0
        # Inputting stage
        if (self.current_step < self.columns):
            input = self.input[self.current_step]
            # Only increase score if push happens in valid area
            # this excludes the columns where the delimiters occur
            if (action[0] and not action[1] and self.current_step != 0 and self.current_step != self.columns-1):
                score += 1
            elif (not action[0] and not action[1] and (self.current_step == 0 or self.current_step == self.columns-1)):
                score += 1
            else:
                score -= 1
        # Outputting stage
        else:
            input = np.array([0] * self.rows)
            if (action[1] and not action[0] and self.memory):
                score += 1
            else:
                score -= 1
            
        # Pop from memory
        if (action[1] and self.memory):
            pop = self.memory.pop()
            self.stdio.append(pop)

        # Check stdio vs true output
        #if (np.all(self.stdio == self.output)):
            #print(self.stdio, self.output)
            #quit()

        # Push to memory if action is set
        if (action[0]):
            self.memory.append(input[2:])

        # Set to done on last step
        done = False
        if (self.current_step >= self.numSteps):
            done = True

        # Increment current step
        self.current_step += 1

        return input, score/self.numSteps, done, "N/A"
