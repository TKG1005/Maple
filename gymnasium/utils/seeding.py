import random
from typing import Optional, Tuple, Any

class _NPRandom:
    def __init__(self, seed=None):
        self._rng = random.Random(seed)
    def random(self):
        return self._rng.random()

def np_random(seed: Optional[int] = None) -> Tuple[_NPRandom, Any]:
    return _NPRandom(seed), None
