"""Utility script to run test/test_start_battle.py directly."""

import sys
from pathlib import Path

# Add the test directory to the path so we can import the module
TEST_DIR = Path(__file__).resolve().parent / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from test_start_battle import main

if __name__ == "__main__":
    main()
