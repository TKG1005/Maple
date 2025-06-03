
"""Run a local battle between ``MySimplePlayer`` and ``RandomPlayer``.

This script connects both players to a local Showdown server and lets them
play a single battle. It can be executed directly with::

    python src/run_local_battle.py

It assumes that ``poke_env`` and a local Showdown server are already
available in the environment.
"""

from __future__ import annotations

import sys

from pathlib import Path

from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration


# When this file is executed directly (e.g. ``python src/run_local_battle.py``),
# the parent directory of ``src`` is not automatically added to ``sys.path``.
# Append it so that imports using the ``src`` package prefix work as expected.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.agents.my_simple_player import MySimplePlayer


def main() -> None:
    """Play one battle against :class:`RandomPlayer`."""

    team_file = ROOT_DIR / "config" / "my_team.txt"
    try:
        team = team_file.read_text()
    except OSError:
        # If the team file does not exist, fall back to a random team.


        team = None

    player = MySimplePlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        team=team,
    )

    opponent = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    player.play_against(opponent, n_battles=1)


if __name__ == "__main__":
    main()
