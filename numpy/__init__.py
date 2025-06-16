"""Minimal NumPy stub with optional delegation to the real package."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SEARCH_PATHS = [p for p in sys.path if Path(p).resolve() != _THIS_DIR.parent]

spec = importlib.machinery.PathFinder.find_spec("numpy", _SEARCH_PATHS)
if spec and spec.origin != __file__:
    module = importlib.util.module_from_spec(spec)
    sys.modules[__name__] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    globals().update(module.__dict__)
else:
    import random

    float32 = float

    class ndarray(list):
        def __add__(self, other):
            if isinstance(other, (int, float)):
                return ndarray([x + other for x in self])
            if isinstance(other, (list, ndarray)):
                return ndarray([x + y for x, y in zip(self, other)])
            return NotImplemented

        __radd__ = __add__

    class Generator:
        """Placeholder for numpy.random.Generator."""

        pass

    class _RandomModule:
        Generator = Generator

        def choice(self, a, size=None, replace=True):
            seq = list(range(a)) if isinstance(a, int) else list(a)
            if size is None:
                return random.choice(seq)
            if replace:
                return [random.choice(seq) for _ in range(size)]
            return random.sample(seq, size)

        def random(self):
            return random.random()

    random = _RandomModule()

    def asarray(obj, dtype=None):
        if isinstance(obj, ndarray):
            return obj
        if isinstance(obj, list):
            return ndarray(obj)
        return ndarray([obj])

    def zeros(shape, dtype=float):
        if isinstance(shape, int):
            return ndarray([dtype(0) for _ in range(shape)])
        return ndarray([[dtype(0) for _ in range(shape[1])] for _ in range(shape[0])])

    def ones(shape, dtype=float):
        if isinstance(shape, int):
            return ndarray([dtype(1) for _ in range(shape)])
        return ndarray([[dtype(1) for _ in range(shape[1])] for _ in range(shape[0])])

    def stack(arrays):
        return ndarray([list(arr) for arr in arrays])

    __all__ = [
        "ndarray",
        "float32",
        "random",
        "asarray",
        "zeros",
        "ones",
        "stack",
        "Generator",
    ]
