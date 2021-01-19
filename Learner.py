from utils import roundedSigmoid
import numpy as np
import pickle

from Program import Program

class Learner:

    def __init__(self):
        self.program = Program()
        self.reset()


    def reset(self):
        self.fitness = None
        self.num_skips = 0


    def act(self, state):
        # NOTE: Actions are defined very narrowly for CartPole:
        #       The value in register 0 is fed through a sigmod
        #       and rounded to either 0 or 1, the two valid
        #       actions for the CartPole environment. This does
        #       not generalize to other environments!!
        # UPDATE: I appended to the function to allow for 
        #       multiple elements. It is still limited
        self.program.execute(state)

        action = self.program.registers
        sig = np.vectorize(roundedSigmoid)
        result = sig(action)

        return result


    def mutate(self):
        self.program.mutate()


    def save(self, name):
        pickle.dump(self, open(name + ".agent", 'wb'))


# Load a saved Learner structure
def loadLearner(fname):
    learner = pickle.load(open(fname, 'rb'))
    return learner
