import numpy as np
from copy import deepcopy
import time

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
    output_folder       = "outputs/",
    env_name            = "output",
    fitness_sharing     = False):
    
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
    Trainer.OUTPUT_FOLDER           = output_folder
    Trainer.ENV_NAME                = env_name
    Trainer.FITNESS_SHARING         = fitness_sharing

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
    MULTI_ELEMENT           = False
    OUTPUT_FOLDER           = "outputs/"
    ENV_NAME                = "output"
    CURRENT_TIME            = int(time.time())
    FITNESS_SHARING         = False

    def __init__(self, env, test_env = False):

        self.learner_pop = []

        if (test_env and Trainer.FITNESS_SHARING):
            print("WARNING: Test environment not used in conjunction with fitness sharing")

        self.env = env
        self.test_env = test_env
        if Trainer.VERBOSE:
            self.write_output("Generation,Average Score,Top Score,Successes\n")

        for i in range(Trainer.POPULATION_SIZE):
            l = Learner()
            self.learner_pop.append(l)

    def write_output(self, content):
        print(content, end="")
        with open(self.OUTPUT_FOLDER+str(self.CURRENT_TIME)+"_"+self.ENV_NAME+".csv", "a") as output:
            output.write(content)

    def evolve(self):
        for i in range(Trainer.NUM_GENERATIONS):
            self.write_output("{},".format(i))

            # Generate Learners to fill the population
            self.generation()

            # Evaluate the Learners' fitness
            self.evaluation()

            # Select the fittest Learners
            self.selection()

        # Testing the fittest Learners
        if (self.test_env):
            self.evaluation(True)


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


    def evaluation(self, test_set = False):
        '''Measures the fitness of all Learners in the population.'''

        scores = []
        successes = 0

        if (Trainer.FITNESS_SHARING):
            scores, successes = self.fitnessSharingEvaluation()
        else:
            for _, learner in enumerate(self.learner_pop):

                # Evaluate the agent in the current task/environment
                self.evaluateLearner(learner, test_set)
                scores.append(learner.fitness)
                successes += learner.successes/len(self.learner_pop)

            if (test_set):
                self.write_output("\n----------\nTest Set Results:\n----------\n")

        if (Trainer.VERBOSE):
            self.write_output("{},{},{}\n".format(np.mean(scores),np.max(scores),successes))

    def fitnessSharingEvaluation(self):
        print("Fitness Sharing")
        collectedScores = []
        env = self.env
        for ep in range(Trainer.NUM_EPISODES_PER_GEN):
            if Trainer.ENV_SEED >= 0:
                env.seed(Trainer.ENV_SEED)
            state = env.reset()
            scores = []
            successes = []
            for _, learner in enumerate(self.learner_pop):
                # TODO Add restart command to all task
                state = env.restart()
                score, success = self.evaluateOnce(learner, state, env)
                scores.append(score)
                successes.append(score)

            meanScore = np.mean(scores)
            collectedScores.append(scores)
            for idx, learner in enumerate(self.learner_pop):
                if learner.fitness is None:
                    learner.fitness = 0
                learner.fitness += 1 if scores[idx] >= meanScore else 0
            # successes /= Trainer.NUM_EPISODES_PER_GEN
        return collectedScores.mean(axis=1), "N/A"

    def evaluateOnce(self, learner, state, env):
        score = 0.0
        success = 0.0
        # Play out the episode
        done = False
        while not done:
            action = learner.act(state.reshape(-1))
            state, reward, done, debug = env.step(action)
            score += reward
        if (score >= 0.999999999):
            success += 1
        return score, success

    def evaluateLearner(self, learner, test_set):
        '''Evaluate a Learner over some number of episodes in a given environment'''
        # Handle Test Set
        if (test_set):
            env = self.test_env
        else:
            env = self.env

        # Skip agents that have already been evaluated, up to MAX_NUM_SKIPS times
        if learner.fitness is not None:
            if learner.num_skips < Trainer.MAX_NUM_SKIPS:
                learner.num_skips += 1
                return
            else:
                learner.num_skips = 0

        # Track scores across episodes
        scores = []
        successes = 0.0

        for ep in range(Trainer.NUM_EPISODES_PER_GEN):

            # Reset the score and environment for this episode
            if Trainer.ENV_SEED >= 0:
                env.seed(Trainer.ENV_SEED)
            state = env.reset()

            score, success = self.evaluateOnce(learner, state, env)
            scores.append(score)
            successes += success

        learner.successes = successes/Trainer.NUM_EPISODES_PER_GEN
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
