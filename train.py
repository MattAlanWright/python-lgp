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

from CopyTask import CopyTask
from TMaze import TMaze
from SeqClassing import SeqClassing
from SeqRecall import SeqRecall

from Program import ConfigureProgram
from Trainer import ConfigureTrainer, Trainer

import os
import sys
import argparse

def run(arguments):

    parser = argparse.ArgumentParser(description='Perform linear GP evolution for a given environment.')

    # Program configuration
    parser.add_argument('--minps', dest='min_prog_size', type=int, help='Minimum number of instructions per Program', default=32)
    parser.add_argument('--maxps', dest='max_prog_size', type=int, help='Maximum number of instructions per Program', default=1024)
    parser.add_argument('--padd', dest='padd', type=float, help='Instruction addition strength', default=0.7)
    parser.add_argument('--pdel', dest='pdel', type=float, help='Instruction deletion strength', default=0.7)
    parser.add_argument('--pmut', dest='pmut', type=float, help='Instruction mutation strength', default=0.7)

    # Trainer configuration
    parser.add_argument('--generations', dest='num_generations', type=int, help='Number of generations over which evolution is performed', default=50)
    parser.add_argument('--pop', dest='population_size', type=int, help='Learner population size', default=200)
    parser.add_argument('--keep', dest='percent_keep', type=float, help='Percentage of surviving Learners', default=0.3)
    parser.add_argument('--fast', dest='fast_mode', type=bool, help='Skip some re-evaluations', default=True)
    parser.add_argument('--skips', dest='num_skips', type=int, help='Number of generations over which to skip re-evaluation', default=3)
    parser.add_argument('--episodes', dest='num_eps_per_gen', type=int, help='Number of episodes over which an agent is evaluated each generation', default=3)
    parser.add_argument('--verbose', dest='verbose', type=bool, help='Do print out info to the console during evolution', default=True)
    parser.add_argument('--agent', dest='agent_save_name', type=str, help='Name under which to save the evolved agent', default="")
    parser.add_argument('--fitness_sharing', dest='fitness_sharing', type=bool, help='Use fitness sharing to evaluate learners', default=False)

    # Environment configuration
    parser.add_argument('--env', dest='env', type=str, help='OpenAI environment', default="CartPole-v1")
    parser.add_argument('--statespace', dest='statespace', type=int, help='Length of flattened state space', default=4)
    args = parser.parse_args(arguments)

    # Default CartPole
    # env = gym.make(args.env)
    # Current functioning tasks
    if args.env == "copytask":
        env = CopyTask(8,8)
    elif args.env == "tmaze":
        args.statespace = 2
        env = TMaze()
        test_env = TMaze(100)
    elif args.env == "seqrecall":
        env = SeqRecall()
        args.statespace = 1
    elif args.env == "seqclass":
        args.statespace = 1
        env = SeqClassing()
    elif args.env == "seqclassconst":
        args.statespace = 1
        env = SeqClassing(7)
    else:
        print("Please state an environment/task:")
        print("copytask,tmaze,seqrecall,seqclass,seqclassconst")
        return

    ConfigureProgram(
        num_inputs      = args.statespace,
        min_prog_size   = args.min_prog_size,
        max_prog_size   = args.max_prog_size,
        p_add           = args.padd,
        p_del           = args.pdel,
        p_mut           = args.pmut)

    ConfigureTrainer(
        num_generations     = args.num_generations,
        population_size     = args.population_size,
        percent_keep        = args.percent_keep,
        fast_mode           = args.fast_mode,
        max_num_skips       = args.num_skips,
        num_eps_per_gen     = args.num_eps_per_gen,
        verbose             = args.verbose,
        agent_save_name     = args.agent_save_name,
        output_folder       = "../lgp-outputs/",
        env_name            = args.env,
        fitness_sharing     = args.fitness_sharing)

    trainer = Trainer(env)
    trainer.evolve()

if __name__ == "__main__":
    run(sys.argv[1:])
