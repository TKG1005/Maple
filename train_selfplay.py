from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor

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
    file_handler = logging.FileHandler(
        log_path / f"train_{timestamp}.log", encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    file_handler.setLevel(logging.DEBUG)
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
from src.algorithms import PPOAlgorithm, ReinforceAlgorithm, compute_gae  # noqa: E402


def init_env(reward: str = "composite", reward_config: str | None = None) -> PokemonEnv:
    """Create :class:`PokemonEnv` for self-play."""
    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
        reward=reward,
        reward_config_path=reward_config,
    )
    return env


def run_episode(
    env: PokemonEnv,
    agent0: RLAgent,
    agent1: RLAgent,
    value_net: torch.nn.Module,
    gamma: float,
    lam: float,
    record_init: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    float,
    tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    dict[str, float],
]:
    """Run one self-play episode and return batch data and total reward.

    Returns
    -------
    batch : dict[str, np.ndarray]
        Collected trajectories for PPO update.
    total_reward : float
        Episode reward summed over steps.
    init_tuple : tuple[np.ndarray, np.ndarray, np.ndarray] | None
        Initial observation, mask and action probabilities for logging.
    sub_totals : dict[str, float]
        Sum of sub reward values from :attr:`env._sub_reward_logs`.
    """

    observations, info, masks = env.reset(return_masks=True)
    obs0 = observations[env.agent_ids[0]]
    obs1 = observations[env.agent_ids[1]]
    mask0, mask1 = masks

    init_tuple: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if info.get("request_teampreview"):
        order0 = agent0.choose_team(obs0)
        order1 = agent1.choose_team(obs1)
        observations, *_ , masks = env.step(
            {env.agent_ids[0]: order0, env.agent_ids[1]: order1}, return_masks=True
        )
        obs0 = observations[env.agent_ids[0]]
        obs1 = observations[env.agent_ids[1]]
        mask0, mask1 = masks

    if record_init:
        init_obs = obs0.copy()
        init_mask = mask0.copy()
        init_probs = agent0.select_action(init_obs, init_mask)
        init_tuple = (init_obs, init_mask, init_probs)

    done = False
    traj: List[dict] = []
    sub_totals: dict[str, float] = {}

    while not done:
        probs0 = agent0.select_action(obs0, mask0)
        probs1 = agent1.select_action(obs1, mask1)
        rng = env.rng
        act0 = int(rng.choice(len(probs0), p=probs0))
        act1 = int(rng.choice(len(probs1), p=probs1))
        logp0 = float(np.log(probs0[act0] + 1e-8))
        val0 = float(value_net(torch.as_tensor(obs0, dtype=torch.float32)).item())

        actions = {env.agent_ids[0]: act0, env.agent_ids[1]: act1}
        observations, rewards, terms, truncs, _, next_masks = env.step(
            actions, return_masks=True
        )
        if hasattr(env, "_sub_reward_logs"):
            logs = env._sub_reward_logs.get(env.agent_ids[0], {})
            for name, val in logs.items():
                sub_totals[name] = sub_totals.get(name, 0.0) + float(val)
        reward = float(rewards[env.agent_ids[0]])
        done = terms[env.agent_ids[0]] or truncs[env.agent_ids[0]]

        traj.append(
            {
                "obs": obs0,
                "action": act0,
                "log_prob": logp0,
                "value": val0,
                "reward": reward,
            }
        )

        obs0 = observations[env.agent_ids[0]]
        obs1 = observations[env.agent_ids[1]]
        mask0, mask1 = next_masks

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
        "rewards": np.array(rewards, dtype=np.float32),
    }

    return batch, sum(rewards), init_tuple, sub_totals


def main(
    *,
    config_path: str = str(ROOT_DIR / "config" / "train_config.yml"),
    episodes: int | None = None,
    save_path: str | None = None,
    tensorboard: bool = False,
    ppo_epochs: int = 1,
    clip_range: float = 0.0,
    gae_lambda: float = 1.0,
    value_coef: float = 0.0,
    entropy_coef: float = 0.0,
    gamma: float = 0.99,
    parallel: int = 1,
    checkpoint_interval: int = 0,
    checkpoint_dir: str = "checkpoints",
    algo: str = "ppo",
    reward: str = "composite",
    reward_config: str | None = str(ROOT_DIR / "config" / "reward.yaml"),
) -> None:
    """Entry point for self-play PPO training.

    Parameters
    ----------
    parallel : int
        Number of environments to run simultaneously.
    reward_config : str | None
        YAML file path for composite reward settings.
    """

    cfg = load_config(config_path)
    episodes = episodes if episodes is not None else int(cfg.get("episodes", 1))
    lr = float(cfg.get("lr", 1e-3))
    gamma = float(cfg.get("gamma", gamma))
    lam = float(cfg.get("gae_lambda", gae_lambda))
    ppo_epochs = int(cfg.get("ppo_epochs", ppo_epochs))
    clip_range = float(cfg.get("clip_range", clip_range))
    value_coef = float(cfg.get("value_coef", value_coef))
    entropy_coef = float(cfg.get("entropy_coef", entropy_coef))
    algo_name = str(cfg.get("algorithm", algo)).lower()
    reward = str(cfg.get("reward", reward))
    reward_config = cfg.get("reward_config", reward_config)
    if reward_config is not None:
        reward_config = str(reward_config)

    writer = SummaryWriter() if tensorboard else None

    envs = [
        init_env(reward=reward, reward_config=reward_config)
        for _ in range(max(1, parallel))
    ]

    ckpt_dir = Path(checkpoint_dir)

    sample_env = envs[0]
    policy_net = PolicyNetwork(
        sample_env.observation_space[sample_env.agent_ids[0]],
        sample_env.action_space[sample_env.agent_ids[0]],
    )
    value_net = ValueNetwork(sample_env.observation_space[sample_env.agent_ids[0]])
    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = optim.Adam(params, lr=lr)

    if algo_name == "ppo":
        algorithm = PPOAlgorithm(
            clip_range=clip_range,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )
    elif algo_name == "reinforce":
        algorithm = ReinforceAlgorithm()
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    agents: list[tuple[RLAgent, RLAgent]] = []
    for env in envs:
        a0 = RLAgent(env, policy_net, value_net, optimizer, algorithm=algorithm)
        a1 = RLAgent(env, policy_net, value_net, optimizer, algorithm=algorithm)
        env.register_agent(a1, env.agent_ids[1])
        agents.append((a0, a1))

    init_obs: np.ndarray | None = None
    init_mask: np.ndarray | None = None
    init_probs: np.ndarray | None = None

    for ep in range(episodes):
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(envs)) as executor:
            futures = [
                executor.submit(
                    run_episode,
                    envs[i],
                    agents[i][0],
                    agents[i][1],
                    value_net,
                    gamma,
                    lam,
                    record_init=(ep == 0 and i == 0),
                )
                for i in range(len(envs))
            ]
            results = [f.result() for f in futures]

        batches = [res[0] for res in results]
        reward_list = [res[1] for res in results]
        sub_logs_list = [res[3] for res in results]

        if ep == 0 and results[0][2] is not None:
            init_obs, init_mask, init_probs = results[0][2]

        combined = {}
        for key in batches[0].keys():
            arrays = [b[key] for b in batches]
            if hasattr(np, "concatenate"):
                combined[key] = np.concatenate(arrays, axis=0)
            else:  # fallback for test stubs
                combined[key] = np.stack([item for arr in arrays for item in arr], axis=0)

        if algo_name == "ppo":
            for i in range(ppo_epochs):
                loss = agents[0][0].update(combined)
                logger.info("Episode %d epoch %d loss %.4f", ep + 1, i + 1, loss)
                if writer:
                    writer.add_scalar("loss", loss, ep * ppo_epochs + i)
        else:
            loss = agents[0][0].update(combined)
            logger.info("Episode %d loss %.4f", ep + 1, loss)
            if writer:
                writer.add_scalar("loss", loss, ep + 1)

        total_reward = sum(reward_list)
        duration = time.perf_counter() - start_time
        logger.info(
            "Episode %d reward %.2f time/episode: %.3f",
            ep + 1,
            total_reward,
            duration,
        )
        if writer:
            writer.add_scalar("reward", total_reward, ep + 1)
            writer.add_scalar("time/episode", duration, ep + 1)
            sub_totals = {}
            for logs in sub_logs_list:
                for name, val in logs.items():
                    sub_totals[name] = sub_totals.get(name, 0.0) + val
            for name, val in sub_totals.items():
                writer.add_scalar(f"sub_reward/{name}", val, ep + 1)

        if checkpoint_interval and (ep + 1) % checkpoint_interval == 0:
            try:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"checkpoint_ep{ep + 1}.pt"
                torch.save(
                    {
                        "policy": policy_net.state_dict(),
                        "value": value_net.state_dict(),
                    },
                    ckpt_path,
                )
                logger.info("Checkpoint saved to %s", ckpt_path)
            except OSError as exc:
                logger.error("Failed to save checkpoint: %s", exc)

    if init_obs is not None and init_mask is not None and init_probs is not None:
        updated_probs = agents[0][0].select_action(init_obs, init_mask)
        diff = updated_probs - init_probs
        logger.info("Initial probs: %s", np.array2string(init_probs, precision=3))
        logger.info("Updated probs: %s", np.array2string(updated_probs, precision=3))
        logger.info("Prob change: %s", np.array2string(diff, precision=3))

    for env in envs:
        env.close()
    if writer:
        writer.close()
    if save_path is not None:
        path = Path(save_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "policy": policy_net.state_dict(),
                    "value": value_net.state_dict(),
                },
                path,
            )
            logger.info("Model saved to %s", path)
        except OSError as exc:
            logger.error("Failed to save model: %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play PPO training script")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="logging level (DEBUG, INFO, etc.)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "config" / "train_config.yml"),
        help="path to YAML config file",
    )
    parser.add_argument("--episodes", type=int, help="number of episodes")
    parser.add_argument(
        "--save", type=str, metavar="FILE", help="path to save model (.pt)"
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="enable TensorBoard logging"
    )
    parser.add_argument(
        "--ppo-epochs", type=int, help="PPO update epochs per episode"
    )
    parser.add_argument("--clip", type=float, help="PPO clipping range")
    parser.add_argument("--gae-lambda", type=float, help="GAE lambda")
    parser.add_argument("--value-coef", type=float, help="value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, help="entropy bonus coefficient")
    parser.add_argument("--gamma", type=float, help="discount factor")
    parser.add_argument("--parallel", type=int, default=1, help="number of parallel environments")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["reinforce", "ppo"],
        default="ppo",
        help="training algorithm (reinforce or ppo)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="save intermediate models every N episodes (0 to disable)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="directory to store checkpoint files",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="composite",
        help="reward function to use (composite)",
    )
    parser.add_argument(
        "--reward-config",
        type=str,
        default=str(ROOT_DIR / "config" / "reward.yaml"),
        help="path to composite reward YAML file",
    )
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")
    setup_logging("logs", vars(args))

    main(
        config_path=args.config,
        episodes=args.episodes,
        save_path=args.save,
        tensorboard=args.tensorboard,
        ppo_epochs=args.ppo_epochs if args.ppo_epochs is not None else 1,
        clip_range=args.clip if args.clip is not None else 0.0,
        gae_lambda=args.gae_lambda if args.gae_lambda is not None else 1.0,
        value_coef=args.value_coef if args.value_coef is not None else 0.0,
        entropy_coef=args.entropy_coef if args.entropy_coef is not None else 0.0,
        gamma=args.gamma if args.gamma is not None else 0.99,
        parallel=args.parallel,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        algo=args.algo,
        reward=args.reward,
        reward_config=args.reward_config,
    )
