from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Repository root path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)

# Ensure bundled poke_env package is importable
POKE_ENV_DIR = ROOT_DIR / "copy_of_poke-env"
if str(POKE_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(POKE_ENV_DIR))

from src.env.wrappers import SingleAgentCompatibilityWrapper  # noqa: E402
from src.env.pokemon_env import PokemonEnv  # noqa: E402
from src.state.state_observer import StateObserver  # noqa: E402
from src.action import action_helper  # noqa: E402


def init_env() -> SingleAgentCompatibilityWrapper:
    """Create :class:`PokemonEnv` wrapped for single-agent use."""

    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
    )
    return SingleAgentCompatibilityWrapper(env)


def main(dry_run: bool = False) -> None:
    """Entry point for RL training script."""

    env = init_env()

    # For dry-run we only initialise the environment
    observation, info = env.reset()
    logger.info("Environment initialised")

    if dry_run:
        env.close()
        logger.info("Dry run complete")
        return

    # TODO: implement training loop

    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="RL training script")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="initialise environment and exit",
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run)
