"""Run a single battle and return the result as JSON serialisable data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time

from src.agents.MapleAgent import MapleAgent
from src.agents.rule_based_player import RuleBasedPlayer
from src.env.pokemon_env import PokemonEnv
from src.state.state_observer import StateObserver
from src.action import action_helper
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

TEAM_FILE = ROOT_DIR / "config" / "my_team.txt"
try:
    TEAM = TEAM_FILE.read_text()
except OSError:
    TEAM = None


def run_single_battle() -> dict:
    """Play one battle using :class:`PokemonEnv` against :class:`RuleBasedPlayer`."""

    opponent = RuleBasedPlayer(
        battle_format="gen9ou",
        server_configuration=LocalhostServerConfiguration,
        team=TEAM,
    )

    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(opponent_player=opponent, state_observer=observer, action_helper=action_helper)
    agent = MapleAgent(env)

    observation, info = env.reset()

    if info.get("request_teampreview"):
        team_order = agent.choose_team(observation)
        observation, mapping, _, _, info = env.step(team_order)
    else:
        battle = next(iter(env._env_player.battles.values()))
        _, mapping = action_helper.get_available_actions_with_details(battle)

    agent.play_until_done(observation, mapping, info)

    battle = next(iter(env._env_player.battles.values()))

    winner = "env" if env._env_player.n_won_battles == 1 else "opponent"
    turns = getattr(battle, "turn", 0)

    return {"winner": winner, "turns": turns}


def main(n: int = 1) -> dict:
    results: List[Dict[str, int | str]] = []
    for _ in tqdm(range(n), desc="Battles"):
        result = run_single_battle()
        results.append(result)

    avg_turns = sum(r["turns"] for r in results) / n if n else 0
    return {"results": results, "average_turns": avg_turns}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run battles locally")
    parser.add_argument("--n", type=int, default=1, help="number of battles")
    args = parser.parse_args()

    result = main(args.n)
    print(json.dumps(result, ensure_ascii=False))
