"""
Based on functions from Joon, tidied up a little.

authors:
    William Tong (wlt2115@columbia.edu)
    Young Joon Kim (yk2611@columbia.edu)
date: March 7, 2020
"""
import random

def sample_idxs(size, split):
    test_len = int(split * size)
    idxs = random.sample(range(size), size)
    test_idxs = idxs[:test_len]
    train_idxs = idxs[test_len:]

    return train_idxs, test_idxs
