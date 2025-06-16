"""Minimal ``numpy.typing`` stub delegating to the real package when present."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SEARCH_PATHS = [p for p in sys.path if Path(p).resolve() != _THIS_DIR.parent.parent]

spec = importlib.machinery.PathFinder.find_spec("numpy.typing", _SEARCH_PATHS)
if spec and spec.origin != __file__:
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    globals().update(module.__dict__)
    sys.modules[__name__] = module
else:
    class NDArray(list):
        """Simple list-based stand-in for numpy.ndarray."""

        pass

    __all__ = ["NDArray"]
