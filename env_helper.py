import gym
from CopyTask import CopyTask
from TMaze import TMaze
from SeqClassing import SeqClassing
from SeqRecall import SeqRecall

def set_env(args):
    if args.env == "cartpole":
        env = gym.make(args.env)
    elif args.env == "copytask":
        #env = CopyTask(8, [10, 20, 50, 100])
        env = CopyTask(3, [100])
    elif args.env == "tmaze":
        args.statespace = 2
        env = TMaze()
        test_env = TMaze(100)
    elif args.env == "seqrecall":
        env = SeqRecall(100,6)
        args.statespace = 1
    elif args.env == "seqclass":
        args.statespace = 1
        env = SeqClassing()
    elif args.env == "seqclassconst":
        args.statespace = 1
        env = SeqClassing(7)
    else:
        print("Please state an environment/task:")
        print("cartpole,copytask,tmaze,seqrecall,seqclass,seqclassconst")
        quit()
    
    return env, args
