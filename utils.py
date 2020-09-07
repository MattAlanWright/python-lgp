import numpy as np

def sigmoid(z):
    '''Logistic function.

    Params:
    z: Single floating value input to the logistic function
    '''
    return 1.0 / (1.0 + np.exp(-z))


def weightedCoinFlip(probability):
    return np.random.uniform(0.0, 1.0) < probability