from typing import Any

ActionType = Any
ObsType = Any

class ParallelEnv:
    def __class_getitem__(cls, item):
        return cls
