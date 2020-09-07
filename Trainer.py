import numpy as np
from copy import deepcopy

from Learner import Learner

class Trainer:

    NUM_GENERATIONS         = 100
    POPULATION_SIZE         = 20
    PERCENT_KEEP            = 0.3
    FAST_MODE               = True
    MAX_NUM_SKIPS           = 3
    NUM_EPISODES_PER_GEN    = 5
    VERBOSE                 = True
    ENV_SEED                = -1
    AGENT_SAVE_NAME         = "top"

    def __init__(self, env):

        # Learner population
        self.learner_pop = []

        # Reference to the environment in which to train
        self.env = env
        
        # Create POPULATION_SIZE new Learners
        for i in range(Trainer.POPULATION_SIZE):
            l = Learner()
            self.learner_pop.append(l)

    
    def evolve(self):

        for i in range(Trainer.NUM_GENERATIONS):

            print("Generation {}".format(i))

            # Generate Learners to fill the population
            self.generation()

            # Evaluate the Learners
            self.evaluation()

            # Select the fittest Learners
            self.selection()


    def generation(self):
        num_old_learners = len(self.learner_pop)
        num_new_learners = Trainer.POPULATION_SIZE - num_old_learners
        for i in range(num_new_learners):

            # Randomly select a Learner from the current population
            l = np.random.choice(self.learner_pop[:num_old_learners])

            # Make a deep copy of the Learner (no old references)
            l_prime = deepcopy(l)

            # Reset new Learner's fitness and num_skips
            l_prime.reset()

            # Mutate the new learner
            l_prime.mutate()

            # Add the learner to the population
            self.learner_pop.append(l_prime)


    def evaluation(self):
        
        scores = []

        # Get fitness of all root Teams
        for i, learner in enumerate(self.learner_pop):

            # Evaluate the agent in the current task/environment
            self.evaluateLearner(learner)
            scores.append(learner.fitness)

        # Print average and top score this generation
        if Trainer.VERBOSE:
            print("    Average score this generation:", int(np.mean(scores)))
            print("    Top score this generation:", int(np.max(scores)))


    def evaluateLearner(self, learner):
        """Evaluate a Learner over some number of episodes in a given environment 'env'
        """

        # Skip agents that have already been evaluated, up to MAX_NUM_SKIPS times
        if learner.fitness is not None:
            if learner.num_skips < Trainer.MAX_NUM_SKIPS:
                learner.num_skips += 1
                return
            else:
                learner.num_skips = 0

        # Track scores across episodes
        scores = []

        for ep in range(Trainer.NUM_EPISODES_PER_GEN):

            # Reset the environment for a new episode
            if Trainer.ENV_SEED >= 0:
                self.env.seed(Trainer.ENV_SEED)

            state = self.env.reset()

            # Reset the score for this episode
            score = 0

            # Loop until the episode is done
            done = False
            while not done:

                # Retrieve the Learner's action
                action = learner.act(state.reshape(-1))

                # Perform action and get next state
                state, reward, done, debug = self.env.step(action)

                # Keep running tally of score
                score += reward

            # Record this episode's score
            scores.append(score)

        # Assign final fitness to Learner
        learner.fitness = np.mean(scores)


    def selection(self):
        """During evolution, after evaluating all Learners, delete the PERCENT_KEEP worst-performing Teams.
        """

        # Sort root Teams from best to worst
        ranked_learners = sorted(self.learner_pop, key=lambda l : l.fitness, reverse=True)

        # Save trainer and top agent so far
        if Trainer.AGENT_SAVE_NAME != "":
            ranked_learners[0].save(Trainer.AGENT_SAVE_NAME)

        # Calculate the number of root Teams to retain
        num_keep = int(Trainer.PERCENT_KEEP * Trainer.POPULATION_SIZE)

        # Save top-performers, let the rest be garbage-collected
        self.learner_pop = ranked_learners[:num_keep]


