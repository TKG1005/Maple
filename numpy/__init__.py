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


class _Random:
    def choice(self, a, size=None, replace=True):
        seq = list(range(a)) if isinstance(a, int) else list(a)
        if size is None:
            return random.choice(seq)
        if replace:
            return [random.choice(seq) for _ in range(size)]
        return random.sample(seq, size)

random = _Random()


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
