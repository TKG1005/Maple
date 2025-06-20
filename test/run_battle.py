"""Run a single battle and return the result as JSON serialisable data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import logging

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency

    def tqdm(data, *args, **kwargs):  # type: ignore[return-type]
        """Fallback if ``tqdm`` is not installed."""
        return data


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)

# Ensure bundled ``poke_env`` package is importable without installation
POKE_ENV_DIR = ROOT_DIR / "copy_of_poke-env"
if str(POKE_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(POKE_ENV_DIR))

from src.agents.MapleAgent import MapleAgent  # noqa: E402
from src.env.pokemon_env import PokemonEnv  # noqa: E402
from src.state.state_observer import StateObserver  # noqa: E402
from src.action import action_helper  # noqa: E402
from src.agents import PolicyNetwork, RLAgent  # noqa: E402
import torch  # noqa: E402
from torch import optim  # noqa: E402

TEAM_FILE = ROOT_DIR / "config" / "my_team.txt"
try:
    TEAM = TEAM_FILE.read_text()
except OSError:
    TEAM = None


def run_single_battle(model_path: str | None = None) -> dict:
    """Play one battle using :class:`PokemonEnv`.

    When ``model_path`` is provided, ``player_0`` will use an :class:`RLAgent`
    loaded from the given file.  Otherwise both players act randomly using
    :class:`MapleAgent`.
    """

    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
    )
    if model_path:
        model = PolicyNetwork(
            env.observation_space[env.agent_ids[0]],
            env.action_space[env.agent_ids[0]],
        )
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        agent0 = RLAgent(env, model, optimizer)
    else:
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
    last_reward = 0.0
    while not done:
        mask0, _ = env.get_action_mask(env.agent_ids[0])
        mask1, _ = env.get_action_mask(env.agent_ids[1])

        action_idx0 = 0
        action_idx1 = 0
        if env._need_action[env.agent_ids[0]]:
            if isinstance(agent0, RLAgent):
                action_idx0 = agent0.act(current_obs0, mask0)
            else:
                action_idx0 = agent0.select_action(current_obs0, mask0)
        if env._need_action[env.agent_ids[1]]:
            action_idx1 = agent1.select_action(current_obs1, mask1)

        observations, rewards, terms, truncs, _ = env.step(
            {"player_0": action_idx0, "player_1": action_idx1}
        )
        last_reward = float(rewards[env.agent_ids[0]])
        current_obs0 = observations[env.agent_ids[0]]
        current_obs1 = observations[env.agent_ids[1]]
        done = terms[env.agent_ids[0]] or truncs[env.agent_ids[0]]

    battle = env.get_current_battle(env.agent_ids[0])
    winner = "env0" if env._env_players["player_0"].n_won_battles == 1 else "env1"
    turns = getattr(battle, "turn", 0)
    reward = last_reward

    env.close()

    logger.info("Battle result winner=%s reward=%.1f turns=%d", winner, reward, turns)

    return {"winner": winner, "turns": turns, "reward": reward}


def main(n: int = 1, model_path: str | None = None) -> dict:
    results: List[Dict[str, float | int | str]] = []
    for _ in tqdm(range(n), desc="Battles"):
        result = run_single_battle(model_path)
        results.append(result)

    avg_turns = sum(r["turns"] for r in results) / n if n else 0
    avg_reward = sum(r["reward"] for r in results) / n if n else 0
    win_rate = (
        sum(1 for r in results if r["winner"] == "env0") / n if n else 0.0
    )
    logger.info(
        "win_rate=%.2f average reward=%.2f average turns=%.2f",
        win_rate,
        avg_reward,
        avg_turns,
    )
    return {
        "results": results,
        "average_turns": avg_turns,
        "average_reward": avg_reward,
        "win_rate": win_rate,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run battles locally")
    parser.add_argument("--n", type=int, default=1, help="number of battles")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="path to trained model (.pt) for player_0",
    )
    args = parser.parse_args()

    result = main(args.n, args.model)
    print(json.dumps(result, ensure_ascii=False))
