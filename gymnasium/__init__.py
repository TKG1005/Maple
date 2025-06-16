import types
from typing import Any

class Space:
    def sample(self) -> Any:
        raise NotImplementedError

    def __class_getitem__(cls, item):
        return cls

class Box(Space):
    def __init__(self, low=0.0, high=1.0, shape=None, dtype=float):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
    def sample(self):
        import numpy as np
        return np.zeros(self.shape, dtype=self.dtype)

class Discrete(Space):
    def __init__(self, n:int):
        self.n = int(n)
    def sample(self):
        import numpy as np
        return int(np.random.choice(self.n))

class Dict(Space):
    def __init__(self, spaces:dict):
        self.spaces = spaces
    def sample(self):
        return {k: s.sample() for k, s in self.spaces.items()}

spaces = types.SimpleNamespace(Box=Box, Dict=Dict, Discrete=Discrete, Space=Space)

class Env:
    metadata = {}
    action_space: Space
    observation_space: Space
    def __init__(self):
        pass

    def __class_getitem__(cls, item):
        return cls
    def reset(self, *, seed=None, options=None):
        return None, {}
    def step(self, action):
        raise NotImplementedError
    def render(self):
        pass
    def close(self):
        pass

class Wrapper(Env):
    def __init__(self, env:Env):
        self.env = env
        self._action_space = getattr(env, 'action_space', None)
        self._observation_space = getattr(env, 'observation_space', None)
    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
    def step(self, action):
        return self.env.step(action)
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    def close(self):
        return self.env.close()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

def register(*args, **kwargs):
    pass

logger = types.SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None)

__all__ = [
    'Env',
    'Wrapper',
    'Space',
    'Box',
    'Discrete',
    'Dict',
    'spaces',
    'register',
    'logger',
]
