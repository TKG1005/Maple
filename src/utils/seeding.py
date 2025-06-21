import random
import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - torch may be optional in some environments
    torch = None


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if hasattr(torch, "cuda"):
                torch.cuda.manual_seed_all(seed)
        except Exception:  # pragma: no cover - torch may not support manual_seed
            pass
