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
    file_handler = logging.FileHandler(log_path / f"eval_{timestamp}.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
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
from src.agents import PolicyNetwork, ValueNetwork, RLAgent  # noqa: E402
import torch  # noqa: E402
from torch import optim  # noqa: E402


def init_env(save_replays: bool | str = False) -> SingleAgentCompatibilityWrapper:
    """Create :class:`PokemonEnv` wrapped for single-agent evaluation."""

    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
        save_replays=save_replays,
    )
    return SingleAgentCompatibilityWrapper(env)


def init_env_multi(save_replays: bool | str = False) -> PokemonEnv:
    """Create :class:`PokemonEnv` for two-agent evaluation."""

    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
        save_replays=save_replays,
    )
    return env


def run_episode(agent: RLAgent) -> tuple[bool, float]:
    """Run one battle and return win flag and total reward."""

    env = agent.env
    obs, info = env.reset()
    if info.get("request_teampreview"):
        team_cmd = agent.choose_team(obs)
        obs, action_mask, _, done, _ = env.step(team_cmd)
    else:
        action_mask, _ = env.env.get_action_mask(env.env.agent_ids[0], with_details=True)
        done = False
    total_reward = 0.0
    while not done:
        action = agent.act(obs, action_mask)
        obs, action_mask, reward, done, _ = env.step(action)
        total_reward += float(reward)
    won = env.env._env_players[env.env.agent_ids[0]].n_won_battles == 1
    return won, total_reward


def run_episode_multi(agent0: RLAgent, agent1: RLAgent) -> tuple[bool, bool, float, float]:
    """Run one battle between two agents and return win flags and rewards."""

    env = agent0.env
    observations, info = env.reset()
    obs0 = observations[env.agent_ids[0]]
    obs1 = observations[env.agent_ids[1]]

    if info.get("request_teampreview"):
        order0 = agent0.choose_team(obs0)
        order1 = agent1.choose_team(obs1)
        observations, *_ = env.step({"player_0": order0, "player_1": order1})
        obs0 = observations[env.agent_ids[0]]
        obs1 = observations[env.agent_ids[1]]

    done = False
    reward0 = 0.0
    reward1 = 0.0
    while not done:
        mask0, _ = env.get_action_mask(env.agent_ids[0], with_details=True)
        mask1, _ = env.get_action_mask(env.agent_ids[1], with_details=True)
        action0 = agent0.act(obs0, mask0) if env._need_action[env.agent_ids[0]] else 0
        action1 = agent1.act(obs1, mask1) if env._need_action[env.agent_ids[1]] else 0
        observations, rewards, terms, truncs, _ = env.step({"player_0": action0, "player_1": action1})
        obs0 = observations[env.agent_ids[0]]
        obs1 = observations[env.agent_ids[1]]
        reward0 += float(rewards[env.agent_ids[0]])
        reward1 += float(rewards[env.agent_ids[1]])
        done = terms[env.agent_ids[0]] or truncs[env.agent_ids[0]]

    win0 = env._env_players[env.agent_ids[0]].n_won_battles == 1
    win1 = env._env_players[env.agent_ids[1]].n_won_battles == 1
    return win0, win1, reward0, reward1


def evaluate_single(model_path: str, n: int = 1, replay_dir: str | bool = "replays") -> None:
    env = init_env(save_replays=replay_dir)
    policy_net = PolicyNetwork(env.observation_space, env.action_space)
    value_net = ValueNetwork(env.observation_space)
    state_dict = torch.load(model_path, map_location="cpu")
    policy_net.load_state_dict(state_dict)
    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    agent = RLAgent(env, policy_net, value_net, optimizer)

    wins = 0
    total_reward = 0.0
    for i in range(n):
        won, reward = run_episode(agent)
        wins += int(won)
        total_reward += reward
        logger.info("Battle %d reward=%.2f win=%s", i + 1, reward, won)

    env.close()
    logger.info("Evaluation finished after %d battles", n)

    win_rate = wins / n if n else 0.0
    avg_reward = total_reward / n if n else 0.0
    logger.info("win_rate: %.2f avg_reward: %.2f", win_rate, avg_reward)


def compare_models(model_a: str, model_b: str, n: int = 1, replay_dir: str | bool = "replays") -> None:
    """Evaluate two models against each other and report win rates."""

    env = init_env_multi(save_replays=replay_dir)

    # Create player_1 agent first so that registration order is correct
    policy1 = PolicyNetwork(env.observation_space[env.agent_ids[1]], env.action_space[env.agent_ids[1]])
    value1 = ValueNetwork(env.observation_space[env.agent_ids[1]])
    state_dict1 = torch.load(model_b, map_location="cpu")
    policy1.load_state_dict(state_dict1)
    params1 = list(policy1.parameters()) + list(value1.parameters())
    opt1 = optim.Adam(params1, lr=1e-3)
    agent1 = RLAgent(env, policy1, value1, opt1)
    env.register_agent(agent1, env.agent_ids[1])

    policy0 = PolicyNetwork(env.observation_space[env.agent_ids[0]], env.action_space[env.agent_ids[0]])
    value0 = ValueNetwork(env.observation_space[env.agent_ids[0]])
    state_dict0 = torch.load(model_a, map_location="cpu")
    policy0.load_state_dict(state_dict0)
    params0 = list(policy0.parameters()) + list(value0.parameters())
    opt0 = optim.Adam(params0, lr=1e-3)
    agent0 = RLAgent(env, policy0, value0, opt0)

    wins0 = 0
    wins1 = 0
    total0 = 0.0
    total1 = 0.0
    for i in range(n):
        win0, win1, reward0, reward1 = run_episode_multi(agent0, agent1)
        wins0 += int(win0)
        wins1 += int(win1)
        total0 += reward0
        total1 += reward1
        logger.info(
            "Battle %d P0_reward=%.2f win=%s P1_reward=%.2f win=%s",
            i + 1,
            reward0,
            win0,
            reward1,
            win1,
        )

    env.close()
    win_rate0 = wins0 / n if n else 0.0
    win_rate1 = wins1 / n if n else 0.0
    avg0 = total0 / n if n else 0.0
    avg1 = total1 / n if n else 0.0
    logger.info("modelA win_rate: %.2f avg_reward: %.2f", win_rate0, avg0)
    logger.info("modelB win_rate: %.2f avg_reward: %.2f", win_rate1, avg1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Evaluate trained RL model")
    parser.add_argument("--model", type=str, help="path to model file (.pt)")
    parser.add_argument("--models", nargs=2, metavar=("A", "B"), help="two model files for head-to-head evaluation")
    parser.add_argument("--n", type=int, default=1, help="number of battles")
    parser.add_argument(
        "--replay-dir",
        type=str,
        default="replays",
        help="directory to save battle replays",
    )
    args = parser.parse_args()

    setup_logging("logs", vars(args))

    if args.models:
        compare_models(args.models[0], args.models[1], args.n, args.replay_dir)
    elif args.model:
        evaluate_single(args.model, args.n, args.replay_dir)
    else:
        parser.error("--model or --models is required")
