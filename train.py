# Learner training script. This script lets you run the TPG agent training routine and
# optionally save lots of different intermediate and final results. It's also an example
# of the steps you need to take if you want to train an Agent on your own outside of
# this script. In general this script is run from the command line. For example:
#
# $ python train.py --env CartPole-v1 --agent_savename cp
#
# will train a TPG Agent to play CartPole-v1 and save the resulting Agent as 'cp.agent'.

import gym
import numpy as np

from Trainer import Trainer

import os
import sys
import argparse

def run(arguments):
    if len(arguments) == 0:
        print("ERROR - No arguments given to main")
        sys.exit(0)

    # Setup the command line parsing to read the environment title
    parser = argparse.ArgumentParser(description='Perform TPG evolution for a given environment.')
    parser.add_argument('--env', dest='env', type=str, help='OpenAI environment', default="CartPole-v1")
    parser.add_argument('--learner', dest='agent_savename', type=str, help='Name under which to save the top Learner object', default="")
    parser.add_argument('--generations', dest='num_generations', type=int, help='Number of generations', default=500)
    parser.add_argument('--episodes', dest='num_episodes', type=int, help='Number of episodes per agent at each generation', default=1)
    parser.add_argument('--pop', dest='r_size', type=int, help='Number of agents (root teams) per generation', default=200)
    parser.add_argument('--seed', dest='seed', type=int, help='Seed for environment', default=-1)
    parser.add_argument('--verbose', dest='verbose', type=bool, help="Print results to the console as we evolve", default=True)
    parser.add_argument('--fast', dest='fast', type=bool, help="Set to True to skip re-evaluating agents", default=True)
    parser.add_argument('--skips', dest='skips', type=int, help='Maximum number of times an agent can skip re-evaluation', default=2)
    args = parser.parse_args(arguments)

    # Get environment details
    env = gym.make(args.env)

    # Create Trainer
    trainer = Trainer(env)

    # Try to generate an agent
    trainer.evolve()

if __name__ == "__main__":
    run(sys.argv[1:])