"""Run a single battle and return the result as JSON serialisable data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency

    def tqdm(data, *args, **kwargs):  # type: ignore[return-type]
        """Fallback if ``tqdm`` is not installed."""
        return data


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure bundled ``poke_env`` package is importable without installation
POKE_ENV_DIR = ROOT_DIR / "copy_of_poke-env"
if str(POKE_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(POKE_ENV_DIR))

from src.agents.MapleAgent import MapleAgent  # noqa: E402
from src.env.pokemon_env import PokemonEnv  # noqa: E402
from src.state.state_observer import StateObserver  # noqa: E402
from src.action import action_helper  # noqa: E402

TEAM_FILE = ROOT_DIR / "config" / "my_team.txt"
try:
    TEAM = TEAM_FILE.read_text()
except OSError:
    TEAM = None


def run_single_battle() -> dict:
    """Play one battle using :class:`PokemonEnv` with two :class:`MapleAgent`s."""

    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
    )
    agent0 = MapleAgent(env)
    agent1 = MapleAgent(env)
    env.register_agent(agent1, "player_1")

    observations, info = env.reset()
    current_obs0 = observations[env.agent_ids[0]]
    current_obs1 = observations[env.agent_ids[1]]

    if info.get("request_teampreview"):
        order0 = agent0.choose_team(current_obs0)
        order1 = agent1.choose_team(current_obs1)
        observations, *_ = env.step({"player_0": order0, "player_1": order1})
        current_obs0 = observations[env.agent_ids[0]]
        current_obs1 = observations[env.agent_ids[1]]

    done = False
    while not done:
        battle0 = env._current_battles[env.agent_ids[0]]
        battle1 = env._current_battles[env.agent_ids[1]]
        mask0, _ = action_helper.get_available_actions_with_details(battle0)
        mask1, _ = action_helper.get_available_actions_with_details(battle1)

        action_idx0 = 0
        action_idx1 = 0
        if env._need_action[env.agent_ids[0]]:
            action_idx0 = agent0.select_action(current_obs0, mask0)
        if env._need_action[env.agent_ids[1]]:
            action_idx1 = agent1.select_action(current_obs1, mask1)

        observations, rewards, terms, truncs, _ = env.step(
            {"player_0": action_idx0, "player_1": action_idx1}
        )
        current_obs0 = observations[env.agent_ids[0]]
        current_obs1 = observations[env.agent_ids[1]]
        done = terms[env.agent_ids[0]] or truncs[env.agent_ids[0]]

    battle = env._current_battles[env.agent_ids[0]]
    winner = "env0" if env._env_players["player_0"].n_won_battles == 1 else "env1"
    turns = getattr(battle, "turn", 0)

    env.close()

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
