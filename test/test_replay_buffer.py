from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.agents.replay_buffer import ReplayBuffer


def test_replay_buffer_capacity():
    capacity = 5
    buf = ReplayBuffer(capacity)
    state = np.zeros(3)
    next_state = np.ones(3)

    for i in range(capacity + 2):
        buf.add(state + i, i, float(i), next_state + i, i % 2 == 0)
        assert len(buf) <= capacity
    assert len(buf) == capacity
