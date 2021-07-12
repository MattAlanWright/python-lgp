import pickle
import sys
import numpy as np
import time
import argparse
from env_helper import *
import glob
from SeqClassing import SeqClassing
sys.path.insert(0, '../')

from Learner import loadLearner

def run(arguments):
    if len(arguments) == 0:
        print("ERROR - No arguments given to main")
        return

    parser = argparse.ArgumentParser(description='Load a linear PG agent into an environment and render the results.')
    parser.add_argument('--env', dest='env', type=str, help='OpenAI environment', default="CartPole-v1")
    parser.add_argument('--agent', dest='agent_fname', type=str, help='Previously saved Agent', default="")
    parser.add_argument('--agent_folder', dest='agent_folder', type=str, help='Previously saved Agent(s) multiple in folder', default="")
    parser.add_argument('--seed', dest='seed', type=int, help='Seed for environment', default=-1)
    args = parser.parse_args(arguments)

    if args.agent_fname == "" and args.agent_folder == "":
        print("No agent name provide!")
        return

    learners = []
    if (args.agent_fname):
        learners.append(loadLearner(args.agent_fname))
    else:
        for fname in glob.glob(args.agent_folder):
            learners.append(loadLearner(fname))

    #env, args = set_env(args)

    # Run through all sizes
    resultsCollection = []
    for hallwaySize in range(10, 20):
        for num_of_ones in range(4, 15):
            results = []
            env = SeqClassing(num_of_ones, hallwaySize)
            results.append(hallwaySize)

            learner = learners[0]
            if args.seed > -1:
                env.seed(args.seed)

            state = env.reset()

            score = 0

            done = False
            while not done:
                if ("render" in dir(env)):
                    env.render()

                # Retrieve the Agent's action
                action = learner.act(state.reshape(-1))

                # Perform action and get next state
                state, reward, done, debug = env.step(action)

                score += reward

                if done:
                    break
                if ("render" in dir(env)):
                    time.sleep(0.01)
            results.append(score)
        resultsCollection.append(results)

    for results in resultsCollection:
        print(','.join(results))

    if ("close" in dir(env)):
        env.close()

if __name__ == "__main__":
    run(sys.argv[1:])
