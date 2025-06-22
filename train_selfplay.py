from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, params: dict[str, object]) -> None:
    """Set up file logging under ``log_dir`` and record parameters."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_path / f"train_{timestamp}.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.info("Run parameters: %s", params)


# Ensure bundled poke_env package is importable
POKE_ENV_DIR = ROOT_DIR / "copy_of_poke-env"
if str(POKE_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(POKE_ENV_DIR))

from src.env.pokemon_env import PokemonEnv  # noqa: E402
from src.state.state_observer import StateObserver  # noqa: E402
from src.action import action_helper  # noqa: E402
from src.agents import (
    PolicyNetwork,
    ValueNetwork,
    RLAgent,
)  # noqa: E402
from src.algorithms import PPOAlgorithm, compute_gae  # noqa: E402


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def init_env() -> PokemonEnv:
    """Create :class:`PokemonEnv` for self-play."""

    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(opponent_player=None, state_observer=observer, action_helper=action_helper)
    return env


def select_action(policy: PolicyNetwork, value: ValueNetwork, obs: np.ndarray, mask: np.ndarray) -> Tuple[int, float, float]:
    """Return sampled action, log probability and value estimate."""

    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    logits = policy(obs_t)
    mask_t = torch.as_tensor(mask, dtype=torch.bool)
    masked = logits.clone()
    if mask_t.any():
        masked[~mask_t] = -float("inf")
    probs = torch.softmax(masked, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = int(dist.sample().item())
    log_prob = float(dist.log_prob(torch.as_tensor(action)))
    val = float(value(obs_t))
    return action, log_prob, val


def process_episode(
    data: List[Tuple[np.ndarray, int, float, float, float]],
    last_value: float,
) -> dict[str, np.ndarray]:
    """Convert stored step data into a training batch."""

    observations = np.stack([d[0] for d in data])
    actions = np.array([d[1] for d in data], dtype=np.int64)
    rewards = [d[2] for d in data]
    log_probs = np.array([d[3] for d in data], dtype=np.float32)
    values = np.array([d[4] for d in data], dtype=np.float32)
    adv = compute_gae(rewards, values, last_value=last_value)
    returns = adv + values
    return {
        "observations": observations,
        "actions": actions,
        "old_log_probs": log_probs,
        "advantages": adv,
        "returns": returns,
        "values": values,
    }


def run_episode(env: PokemonEnv, policy: PolicyNetwork, value: ValueNetwork) -> dict[str, np.ndarray]:
    """Play one self-play episode and return training batch."""

    agent_id0, agent_id1 = env.agent_ids
    observations, info = env.reset()
    obs0 = observations[agent_id0]
    obs1 = observations[agent_id1]
    if info.get("request_teampreview"):
        order0 = "team 0"
        order1 = "team 0"
        observations, *_ = env.step({agent_id0: order0, agent_id1: order1})
        obs0 = observations[agent_id0]
        obs1 = observations[agent_id1]

    buf0: List[Tuple[np.ndarray, int, float, float, float]] = []
    buf1: List[Tuple[np.ndarray, int, float, float, float]] = []
    done = False
    while not done:
        mask0, _ = env.get_action_mask(agent_id0, with_details=True)
        mask1, _ = env.get_action_mask(agent_id1, with_details=True)
        a0, logp0, val0 = select_action(policy, value, obs0, mask0)
        a1, logp1, val1 = select_action(policy, value, obs1, mask1)
        observations, rewards, terms, truncs, _ = env.step({agent_id0: a0, agent_id1: a1})
        r0 = float(rewards[agent_id0])
        r1 = float(rewards[agent_id1])
        buf0.append((obs0, a0, r0, logp0, val0))
        buf1.append((obs1, a1, r1, logp1, val1))
        obs0 = observations[agent_id0]
        obs1 = observations[agent_id1]
        done = terms[agent_id0] or truncs[agent_id0]

    last_val0 = float(value(torch.as_tensor(obs0, dtype=torch.float32)))
    last_val1 = float(value(torch.as_tensor(obs1, dtype=torch.float32)))
    batch0 = process_episode(buf0, last_val0)
    batch1 = process_episode(buf1, last_val1)
    batch = {k: np.concatenate([batch0[k], batch1[k]]) for k in batch0}
    return batch


def main(episodes: int = 1, ppo_epochs: int = 4, lr: float = 1e-3, tensorboard: bool = False, save_path: str | None = None) -> None:
    env = init_env()
    obs_space = env.observation_space[env.agent_ids[0]]
    act_space = env.action_space[env.agent_ids[0]]
    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)
    params = list(policy.parameters()) + list(value.parameters())
    optimizer = optim.Adam(params, lr=lr)
    algorithm = PPOAlgorithm()
    agent0 = RLAgent(env, policy, value, optimizer, algorithm=algorithm)
    agent1 = RLAgent(env, policy, value, optimizer, algorithm=algorithm)
    env.register_agent(agent0, env.agent_ids[0])
    env.register_agent(agent1, env.agent_ids[1])

    writer = SummaryWriter() if tensorboard else None
    global_step = 0

    for ep in range(episodes):
        batch = run_episode(env, policy, value)
        for epoch in range(ppo_epochs):
            loss = algorithm.update(policy, optimizer, batch)
            if writer:
                writer.add_scalar("loss", loss, global_step)
            global_step += 1
        logger.info("Episode %d complete", ep + 1)

    env.close()
    if writer:
        writer.close()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": policy.state_dict(), "value": value.state_dict()}, path)
        logger.info("Model saved to %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Self-play training with PPO")
    parser.add_argument("--episodes", type=int, default=1, help="number of episodes")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO update epochs per episode")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--tensorboard", action="store_true", help="enable TensorBoard logging")
    parser.add_argument("--save", type=str, metavar="FILE", help="path to save trained model")
    args = parser.parse_args()
    setup_logging("logs", vars(args))
    main(args.episodes, args.ppo_epochs, args.lr, args.tensorboard, args.save)

