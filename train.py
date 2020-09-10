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

    parser = argparse.ArgumentParser(description='Perform linear GP evolution for a given environment.')
    parser.add_argument('--minps', dest='min_prog_size', type=int, help='Minimum number of instructions per Program', default=32)
    parser.add_argument('--maxps', dest='max_prog_size', type=int, help='Maximum number of instructions per Program', default=1024)
    parser.add_argument('--padd', dest='padd', type=float, help='Instruction addition strength', default=0.7)
    parser.add_argument('--pdel', dest='pdel', type=float, help='Instruction deletion strength', default=0.7)
    parser.add_argument('--pmut', dest='pmut', type=float, help='Instruction mutation strength', default=0.7)
    parser.add_argument('--env', dest='env', type=str, help='OpenAI environment', default="CartPole-v1")
    parser.add_argument('--statespace', dest='statespace', type=int, help='Length of flattened state space', default=4)
    args = parser.parse_args(arguments)
    
    if args.env != "CartPole-v0" && args.env != "CartPole-v1":
        print("Woops! So far this module only works in the CartPole environment!")
        return
    
    if args.statespace != 4:
        print("Woops! So far this module only works in the CartPole environment with a statespace size of 4!")
        return

    ConfigureProgram(
        num_inputs      = args.statespace,
        min_prog_size   = args.min_prog_size,
        max_prog_size   = args.max_prog_size,
        p_add           = args.padd,
        p_del           = args.pdel,
        p_mut           = args.pmut)

    env = gym.make(args.env)
    trainer = Trainer(env)
    trainer.evolve()

if __name__ == "__main__":
    run(sys.argv[1:])
    
