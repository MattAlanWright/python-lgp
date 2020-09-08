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

from Program import ConfigureProgram
from Trainer import Trainer

import os
import sys
import argparse

def run(arguments):

    # Setup the command line parsing to read the environment title
    parser = argparse.ArgumentParser(description='Perform linear GP evolution for a given environment.')
    parser.add_argument('--env', dest='env', type=str, help='OpenAI environment', default="CartPole-v1")
    parser.add_argument('--statespace', dest='statespace', type=int, help='Length of flattened state space', default=4)
    args = parser.parse_args(arguments)

    ConfigureProgram(num_inputs=args.statespace)

    # Get environment details
    env = gym.make(args.env)

    # Create Trainer
    trainer = Trainer(env)

    # Try to generate an agent
    trainer.evolve()

if __name__ == "__main__":
    run(sys.argv[1:])