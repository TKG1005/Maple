from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Repository root path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, params: dict[str, object]) -> None:
    """Set up file logging and write parameters."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_path / f"eval_{timestamp}.log", encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logging.info("Run parameters: %s", params)


# Ensure bundled poke_env package is importable
POKE_ENV_DIR = ROOT_DIR / "copy_of_poke-env"
if str(POKE_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(POKE_ENV_DIR))

from src.env.wrappers import SingleAgentCompatibilityWrapper  # noqa: E402
from src.env.pokemon_env import PokemonEnv  # noqa: E402
from src.state.state_observer import StateObserver  # noqa: E402
from src.action import action_helper  # noqa: E402
from src.agents import PolicyNetwork, RLAgent  # noqa: E402
from src.utils import seed_everything  # noqa: E402
import torch  # noqa: E402
from torch import optim  # noqa: E402


def init_env(
    seed: int | None = None, battle_seed: int | None = None
) -> SingleAgentCompatibilityWrapper:
    """Create :class:`PokemonEnv` wrapped for single-agent evaluation."""

    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
        seed=seed,
        battle_seed=battle_seed,
    )
    return SingleAgentCompatibilityWrapper(env)


def run_episode(agent: RLAgent, *, seed: int | None = None) -> tuple[bool, float]:
    """Run one battle and return win flag and total reward."""

    env = agent.env
    obs, info = env.reset(seed=seed)
    if info.get("request_teampreview"):
        team_cmd = agent.choose_team(obs)
        obs, action_mask, _, done, _ = env.step(team_cmd)
    else:
        action_mask, _ = env.env.get_action_mask(
            env.env.agent_ids[0], with_details=True
        )
        done = False
    total_reward = 0.0
    while not done:
        action = agent.act(obs, action_mask)
        obs, action_mask, reward, done, _ = env.step(action)
        total_reward += float(reward)
    won = env.env._env_players[env.env.agent_ids[0]].n_won_battles == 1
    return won, total_reward


def main(
    model_path: str,
    n: int = 1,
    *,
    seed: int | None = None,
    battle_seed: int | None = None,
) -> None:
    if seed is not None:
        seed_everything(seed)
    env = init_env(seed=seed, battle_seed=battle_seed)
    model = PolicyNetwork(env.observation_space, env.action_space)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    agent = RLAgent(env, model, optimizer)

    wins = 0
    total_reward = 0.0
    for i in range(n):
        won, reward = run_episode(agent, seed=seed)
        wins += int(won)
        total_reward += reward
        logger.info("Battle %d reward=%.2f win=%s", i + 1, reward, won)

    env.close()
    logger.info("Evaluation finished after %d battles", n)

    win_rate = wins / n if n else 0.0
    avg_reward = total_reward / n if n else 0.0
    logger.info("win_rate: %.2f avg_reward: %.2f", win_rate, avg_reward)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Evaluate trained RL model")
    parser.add_argument(
        "--model", type=str, required=True, help="path to model file (.pt)"
    )
    parser.add_argument("--n", type=int, default=1, help="number of battles")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--battle-seed",
        type=int,
        default=None,
        help="deterministic battle seed for Showdown",
    )
    args = parser.parse_args()

    setup_logging("logs", vars(args))

    main(args.model, args.n, seed=args.seed, battle_seed=args.battle_seed)
