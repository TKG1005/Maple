import random
import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch may not be installed
    torch = None

from src.utils import seed_everything


def test_seed_everything_reproducible():
    if not hasattr(np.random, "seed"):
        pytest.skip("numpy stub does not support seeding")
    seed_everything(123)
    a1 = random.random()
    b1 = np.random.rand()
    if torch is None:
        pytest.skip("torch not available")
    c1 = float(torch.rand(1))

    seed_everything(123)
    assert random.random() == a1
    assert np.random.rand() == b1
    assert float(torch.rand(1)) == c1
