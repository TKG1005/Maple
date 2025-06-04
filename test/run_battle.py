"""Run a single battle and return the result as JSON serialisable data."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.rule_based_player import RuleBasedPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

TEAM_FILE = ROOT_DIR / "config" / "my_team.txt"
try:
    TEAM = TEAM_FILE.read_text()
except OSError:
    TEAM = None


async def run_single_battle() -> dict:
    player_1 = RuleBasedPlayer(
        battle_format="gen9ou",
        server_configuration=LocalhostServerConfiguration,
        log_level=logging.DEBUG,
        team=TEAM,
    )
    player_2 = RuleBasedPlayer(
        battle_format="gen9ou",
        server_configuration=LocalhostServerConfiguration,
        team=TEAM,
    )

    await asyncio.gather(
        player_1.send_challenges(player_2.username, n_challenges=1, to_wait=player_2.ps_client.logged_in),
        player_2.accept_challenges(player_1.username, n_challenges=1)
    )

    winner = "p1" if player_1.n_won_battles == 1 else "p2"
    battle = next(iter(player_1.battles.values()), None)
    turns = getattr(battle, "turn", 0)

    return {"winner": winner, "turns": turns}


async def main(n: int = 1) -> dict:
    results: List[Dict[str, int | str]] = []
    for _ in tqdm(range(n), desc="Battles"):
        result = await run_single_battle()
        results.append(result)

    avg_turns = sum(r["turns"] for r in results) / n if n else 0
    return {"results": results, "average_turns": avg_turns}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run battles locally")
    parser.add_argument("--n", type=int, default=1, help="number of battles")
    args = parser.parse_args()

    result = asyncio.run(main(args.n))
    print(json.dumps(result, ensure_ascii=False))
