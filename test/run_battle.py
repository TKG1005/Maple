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
import logging

from types import SimpleNamespace

import numpy as np
import gymnasium as gym

from src.agents.MapleAgent import MapleAgent
from src.agents.maple_agent_player import MapleAgentPlayer
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
    """Play one battle using :class:`PokemonEnv` with two :class:`MapleAgent`s."""

    dummy_env = SimpleNamespace(
        rng=np.random.default_rng(), action_space=gym.spaces.Discrete(10)
    )
    opponent_agent = MapleAgent(dummy_env)
    opponent = MapleAgentPlayer(
        opponent_agent,
        battle_format="gen9bssregi",
        server_configuration=LocalhostServerConfiguration,
        team=TEAM,
        log_level=logging.DEBUG,
    )

    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=opponent,
        state_observer=observer,
        action_helper=action_helper,
    )
    agent = MapleAgent(env)

    observations, info = env.reset()
    current_obs = observations[env.agent_ids[0]]

    if info.get("request_teampreview"):
        team_order = agent.choose_team(current_obs)
        observations, *_ = env.step({"player_0": team_order})
        current_obs = observations[env.agent_ids[0]]

    done = False
    while not done:
        battle = env._current_battle
        mask, _ = action_helper.get_available_actions_with_details(battle)
        action_idx = agent.select_action(current_obs, mask)
        observations, rewards, terms, truncs, _ = env.step({"player_0": action_idx})
        current_obs = observations[env.agent_ids[0]]
        done = terms[env.agent_ids[0]] or truncs[env.agent_ids[0]]

    battle = env._current_battle
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
