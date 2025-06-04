"""Run multiple random battles and report statistics."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from statistics import mean

from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from tqdm import tqdm


async def run_single_battle() -> dict:
    """Play one battle between two random players and return its result."""
    player_1 = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
    )
    player_2 = RandomPlayer(
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


async def main(n: int) -> dict:
    """Run ``n`` battles sequentially and return aggregated results."""
    results = []
    failures = 0
    for _ in tqdm(range(n), desc="Battles"):
        try:
            result = await run_single_battle()
            logging.info(json.dumps(result, ensure_ascii=False))
            results.append(result)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logging.error("battle failed: %s", exc)
            failures += 1

    avg_turns = mean([r["turns"] for r in results]) if results else 0.0
    summary = {"average_turns": avg_turns, "n_failures": failures}
    logging.info(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random Pokémon battles")
    parser.add_argument("--n", type=int, default=1, help="Number of battles to run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = asyncio.run(main(args.n))
    print(json.dumps(result, ensure_ascii=False))
