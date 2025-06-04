"""Run a single battle and return the result as JSON serialisable data."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.rule_based_player import RuleBasedPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration


async def main() -> dict:
    player_1 = RuleBasedPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        log_level=logging.DEBUG
    )
    player_2 = RuleBasedPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    await asyncio.gather(
        player_1.send_challenges(player_2.username, n_challenges=1),
        player_2.accept_challenges(player_1.username, n_challenges=1),
    )

    winner = "p1" if player_1.n_won_battles == 1 else "p2"
    battle = next(iter(player_1.battles.values()), None)
    turns = getattr(battle, "turn", 0)

    return {"winner": winner, "turns": turns}


if __name__ == "__main__":
    result = asyncio.run(main())
    print(json.dumps(result, ensure_ascii=False))
