import numpy as np
from copy import deepcopy

from Learner import Learner

def ConfigureTrainer(
    num_generations     = 20,
    population_size     = 100,
    percent_keep        = 0.3,
    fast_mode           = True,
    max_num_skips       = 3,
    num_eps_per_gen     = 5,
    verbose             = True,
    env_seed            = -1,
    agent_save_name     = "",
    multi_element       = False):
    
    if percent_keep < 0.1 or percent_keep > 0.9:
        print("Invalid percent_keep {}, must be between 0.1 and 0.9. Setting to 0.3.".format(percent_keep))
        percent_keep = 0.3
        
    Trainer.NUM_GENERATIONS          = num_generations
    Trainer.POPULATION_SIZE         = population_size
    Trainer.PERCENT_KEEP            = percent_keep
    Trainer.FAST_MODE               = fast_mode
    Trainer.MAX_NUM_SKIPS           = max_num_skips
    Trainer.NUM_EPISODES_PER_GEN    = num_eps_per_gen
    Trainer.VERBOSE                 = verbose
    Trainer.ENV_SEED                = env_seed
    Trainer.AGENT_SAVE_NAME         = agent_save_name
    Trainer.MULTI_ELEMENT           = multi_element

class Trainer:

    NUM_GENERATIONS         = 100
    POPULATION_SIZE         = 20
    PERCENT_KEEP            = 0.3
    FAST_MODE               = True
    MAX_NUM_SKIPS           = 3
    NUM_EPISODES_PER_GEN    = 5
    VERBOSE                 = True
    ENV_SEED                = -1
    AGENT_SAVE_NAME         = ""
    MULTI_ELEMENT             = False

    def __init__(self, env):

        self.learner_pop = []

        self.env = env

        for i in range(Trainer.POPULATION_SIZE):
            l = Learner()
            self.learner_pop.append(l)


    def evolve(self):
        for i in range(Trainer.NUM_GENERATIONS):

            print("Generation {}".format(i))

            # Generate Learners to fill the population
            self.generation()

            # Evaluate the Learners' fitness
            self.evaluation()

            # Select the fittest Learners
            self.selection()


    def generation(self):
        num_old_learners = len(self.learner_pop)
        num_new_learners = Trainer.POPULATION_SIZE - num_old_learners
        for _ in range(num_new_learners):

            # Randomly select a Learner from the current population
            l = np.random.choice(self.learner_pop[:num_old_learners])

            # Make a deep copy of the Learner (copy structures, not references)
            l_prime = deepcopy(l)

            # Reset new Learner's fitness and num_skips
            l_prime.reset()

            # Mutate the new learner
            l_prime.mutate()

            # Add the learner to the population
            self.learner_pop.append(l_prime)


    def evaluation(self):
        '''Measures the fitness of all Learners in the population.'''

        scores = []

        for _, learner in enumerate(self.learner_pop):

            # Evaluate the agent in the current task/environment
            self.evaluateLearner(learner)
            scores.append(learner.fitness)

        if Trainer.VERBOSE:
            print("    Average score this generation:", int(np.mean(scores)))
            print("    Top score this generation:", int(np.max(scores)))


    def evaluateLearner(self, learner):
        '''Evaluate a Learner over some number of episodes in a given environment'''

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

            # Reset the score and environment for this episode
            if Trainer.ENV_SEED >= 0:
                self.env.seed(Trainer.ENV_SEED)
            state = self.env.reset()
            score = 0

            # Play out the episode
            done = False
            # For copy task run each column as a step
            # Then run the same number of times for the output
            while not done:

                action = learner.act(state.reshape(-1))
                if (self.MULTI_ELEMENT == False):
                    action = action[0]

                state, reward, done, debug = self.env.step(action)
                score += reward

            #print("Score", score)
            scores.append(score)

        learner.fitness = np.mean(scores)


    def selection(self):
        '''During evolution, after evaluating all Learners, delete the PERCENT_KEEP worst-performing Learners.'''

        # Sort Learners from fittest to least fit
        ranked_learners = sorted(self.learner_pop, key=lambda l : l.fitness, reverse=True)

        if Trainer.AGENT_SAVE_NAME != "":
            ranked_learners[0].save(Trainer.AGENT_SAVE_NAME)

        num_surviving_learners = int(Trainer.PERCENT_KEEP * Trainer.POPULATION_SIZE)

        # Save top-performers, let the rest be garbage-collected
        self.learner_pop = ranked_learners[:num_surviving_learners]
