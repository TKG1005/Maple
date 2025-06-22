from types import ModuleType, SimpleNamespace
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

np_stub = ModuleType("numpy")
np_stub.float32 = "float32"
np_stub.int8 = "int8"
np_stub.ndarray = list
np_stub.random = SimpleNamespace(default_rng=lambda seed=None: None)
np_stub.zeros = lambda shape, dtype=None: [0.0] * shape
np_stub.array = lambda data, dtype=None: list(data)

def asarray(data, dtype=None):
    return list(data)


def zeros_like(a, dtype=None):
    return [0.0 for _ in a]


def allclose(a, b, atol=1e-8, rtol=1e-5):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(x - y) > atol + rtol * abs(y):
            return False
    return True


np_stub.asarray = asarray
np_stub.zeros_like = zeros_like
np_stub.allclose = allclose
sys.modules.setdefault("numpy", np_stub)
sys.modules.setdefault("numpy.random", np_stub.random)
np = np_stub

# Create a minimal torch stub so that importing algorithms does not fail
torch_stub = ModuleType("torch")
torch_stub.nn = ModuleType("torch.nn")
torch_stub.optim = ModuleType("torch.optim")
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.nn", torch_stub.nn)
sys.modules.setdefault("torch.optim", torch_stub.optim)

from src.algorithms.gae import compute_gae


def test_compute_gae_basic():
    rewards = [1.0, 1.0, 1.0]
    values = [0.0, 0.0, 0.0]
    adv = compute_gae(rewards, values, gamma=1.0, lam=1.0)
    assert np.allclose(adv, [3.0, 2.0, 1.0])
