import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import yaml
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

# Repository root path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, params: dict[str, object]) -> None:
    """Set up file logging under ``log_dir`` and record parameters."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_path / f"train_{timestamp}.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.info("Run parameters: %s", params)


def load_config(path: str) -> dict:
    """Load training configuration from a YAML file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError("YAML root must be a mapping")
        return data
    except FileNotFoundError:
        logger.warning("Config file %s not found, using defaults", path)
        return {}


# Ensure bundled poke_env package is importable
POKE_ENV_DIR = ROOT_DIR / "copy_of_poke-env"
if str(POKE_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(POKE_ENV_DIR))

from src.env.pokemon_env import PokemonEnv  # noqa: E402
from src.state.state_observer import StateObserver  # noqa: E402
from src.action import action_helper  # noqa: E402
from src.agents import PolicyNetwork, ValueNetwork, RLAgent  # noqa: E402
from src.algorithms import PPOAlgorithm, compute_gae  # noqa: E402


def init_env() -> PokemonEnv:
    """Create :class:`PokemonEnv` for self-play."""
    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
    )
    return env


def main(
    *,
    config_path: str = str(ROOT_DIR / "config" / "train_config.yml"),
    episodes: int | None = None,
    save_path: str | None = None,
    tensorboard: bool = False,
    ppo_epochs: int = 4,
) -> None:
    """Entry point for self-play PPO training."""

    cfg = load_config(config_path)
    episodes = episodes if episodes is not None else int(cfg.get("episodes", 1))
    lr = float(cfg.get("lr", 1e-3))
    gamma = float(cfg.get("gamma", 0.99))
    lam = float(cfg.get("gae_lambda", 0.95))
    ppo_epochs = int(cfg.get("ppo_epochs", ppo_epochs))

    writer = SummaryWriter() if tensorboard else None

    env = init_env()

    policy_net = PolicyNetwork(env.observation_space[env.agent_ids[0]], env.action_space[env.agent_ids[0]])
    value_net = ValueNetwork(env.observation_space[env.agent_ids[0]])
    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = optim.Adam(params, lr=lr)
    algorithm = PPOAlgorithm()

    agent0 = RLAgent(env, policy_net, value_net, optimizer, algorithm=algorithm)
    agent1 = RLAgent(env, policy_net, value_net, optimizer, algorithm=algorithm)
    env.register_agent(agent1, env.agent_ids[1])

    for ep in range(episodes):
        start_time = time.perf_counter()
        observations, info = env.reset()
        obs0 = observations[env.agent_ids[0]]
        obs1 = observations[env.agent_ids[1]]

        if info.get("request_teampreview"):
            order0 = agent0.choose_team(obs0)
            order1 = agent1.choose_team(obs1)
            observations, *_ = env.step({env.agent_ids[0]: order0, env.agent_ids[1]: order1})
            obs0 = observations[env.agent_ids[0]]
            obs1 = observations[env.agent_ids[1]]

        done = False
        traj: List[dict] = []

        while not done:
            mask0, _ = env.get_action_mask(env.agent_ids[0], with_details=True)
            mask1, _ = env.get_action_mask(env.agent_ids[1], with_details=True)

            probs0 = agent0.select_action(obs0, mask0)
            probs1 = agent1.select_action(obs1, mask1)
            rng = env.rng
            act0 = int(rng.choice(len(probs0), p=probs0))
            act1 = int(rng.choice(len(probs1), p=probs1))
            logp0 = float(np.log(probs0[act0] + 1e-8))
            val0 = float(value_net(torch.as_tensor(obs0, dtype=torch.float32)).item())

            actions = {env.agent_ids[0]: act0, env.agent_ids[1]: act1}
            observations, rewards, terms, truncs, _ = env.step(actions)
            reward = float(rewards[env.agent_ids[0]])
            done = terms[env.agent_ids[0]] or truncs[env.agent_ids[0]]

            traj.append({
                "obs": obs0,
                "action": act0,
                "log_prob": logp0,
                "value": val0,
                "reward": reward,
            })

            obs0 = observations[env.agent_ids[0]]
            obs1 = observations[env.agent_ids[1]]

        rewards = [t["reward"] for t in traj]
        values = [t["value"] for t in traj]
        adv = compute_gae(rewards, values, gamma=gamma, lam=lam)
        returns = [a + v for a, v in zip(adv, values)]

        batch = {
            "observations": np.stack([t["obs"] for t in traj]),
            "actions": np.array([t["action"] for t in traj], dtype=np.int64),
            "old_log_probs": np.array([t["log_prob"] for t in traj], dtype=np.float32),
            "advantages": np.array(adv, dtype=np.float32),
            "returns": np.array(returns, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
        }

        for i in range(ppo_epochs):
            loss = agent0.update(batch)
            if writer:
                writer.add_scalar("loss", loss, ep * ppo_epochs + i)

        total_reward = sum(rewards)
        duration = time.perf_counter() - start_time
        logger.info(
            "Episode %d reward %.2f time/episode: %.3f", ep + 1, total_reward, duration
        )
        if writer:
            writer.add_scalar("reward", total_reward, ep + 1)
            writer.add_scalar("time/episode", duration, ep + 1)

    env.close()
    if writer:
        writer.close()
    if save_path is not None:
        path = Path(save_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(policy_net.state_dict(), path)
            logger.info("Model saved to %s", path)
        except OSError as exc:
            logger.error("Failed to save model: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Self-play PPO training script")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "config" / "train_config.yml"),
        help="path to YAML config file",
    )
    parser.add_argument("--episodes", type=int, default=1, help="number of episodes")
    parser.add_argument("--save", type=str, metavar="FILE", help="path to save model (.pt)")
    parser.add_argument("--tensorboard", action="store_true", help="enable TensorBoard logging")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO update epochs per episode")
    args = parser.parse_args()

    setup_logging("logs", vars(args))

    main(
        config_path=args.config,
        episodes=args.episodes,
        save_path=args.save,
        tensorboard=args.tensorboard,
        ppo_epochs=args.ppo_epochs,
    )
