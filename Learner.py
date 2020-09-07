from utils import sigmoid
import pickle

from Program import Program

class Learner:

    def __init__(self):
        # Initialize a random Program
        self.program = Program()

        self.reset()

    
    def reset(self):
        # Learners are assigned a fitness during evolution
        self.fitness = None

        # Number of times a Learner has skipped fitness re-evaluation
        self.num_skips = 0


    def act(self, state):
        # NOTE: Actions are defined very narrowly for CartPole:
        #       The value in register 0 is fed through a sigmod
        #       and rounded to either 0 or 1, the two valid
        #       actions for the CartPole environment. This does
        #       not generalize to other environments!!
        self.program.execute(state)

        r0 = self.program.registers[0]
        action = int(round(sigmoid(r0)))

        return action


    def mutate(self):
        self.program.mutate()


    def save(self, name):
        pickle.dump(self, open(name + ".agent", 'wb'))


# Load a saved Learner structure
def loadLearner(fname):
    learner = pickle.load(open(fname, 'rb'))
    return learner
