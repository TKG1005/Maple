"""Minimal battle client using poke-env.

Defines :class:`SimpleAgent`, a lightweight player that selects random moves.
Executing this script only creates an instance of the agent and prints the
loaded team text. No battle is started.
"""

from __future__ import annotations

import argparse
from pathlib import Path


class SimpleAgent:
    """Wrap :class:`poke_env.player.Player` with lazy imports."""

    def __init__(self, team: str | None = None) -> None:
        from poke_env.player import Player

        class _InnerPlayer(Player):
            def choose_move(self, battle):  # pragma: no cover - simple random
                return self.choose_random_move(battle)

        self._player = _InnerPlayer(team=team, battle_format="gen9randombattle")
        self.team = team

    def reset_battles(self) -> None:
        self._player.reset_battles()


def main() -> None:
    parser = argparse.ArgumentParser(description="Instantiate a SimpleAgent")
    parser.add_argument("--team", type=Path, default=None, help="Path to team file")
    args = parser.parse_args()

    team_text = args.team.read_text() if args.team else None
    agent = SimpleAgent(team=team_text)
    print(agent.team)


if __name__ == "__main__":
    main()
