from .. import random as _random
Generator = _random.Generator
choice = _random.choice
random = _random.random

def default_rng(seed=None):
    return Generator()

__all__ = ['Generator', 'choice', 'random', 'default_rng']
