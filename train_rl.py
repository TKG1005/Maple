from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Repository root path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)

# Ensure bundled poke_env package is importable
POKE_ENV_DIR = ROOT_DIR / "copy_of_poke-env"
if str(POKE_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(POKE_ENV_DIR))

from src.env.wrappers import SingleAgentCompatibilityWrapper  # noqa: E402
from src.env.pokemon_env import PokemonEnv  # noqa: E402
from src.state.state_observer import StateObserver  # noqa: E402
from src.action import action_helper  # noqa: E402
from src.agents import RLAgent, PolicyNetwork, ReplayBuffer  # noqa: E402
import numpy as np
import torch
import torch.nn as nn


def init_env() -> SingleAgentCompatibilityWrapper:
    """Create :class:`PokemonEnv` wrapped for single-agent use."""

    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
    )
    return SingleAgentCompatibilityWrapper(env)


def select_mask(env: SingleAgentCompatibilityWrapper) -> np.ndarray:
    """Return the current action mask from the underlying environment."""
    battle = env.env._current_battles[env.env.agent_ids[0]]
    mask, _ = action_helper.get_available_actions_with_details(battle)
    return mask


def choose_action(
    agent: RLAgent,
    obs: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
) -> int:
    """Epsilon-greedy action selection."""
    rng = getattr(agent.env, "rng", np.random.default_rng())
    if rng.random() < epsilon:
        valid = [i for i, f in enumerate(mask) if f]
        if not valid:
            valid = list(range(agent.env.action_space.n))
        return int(rng.choice(valid))
    probs = agent.select_action(obs, mask)
    return int(np.argmax(probs))


def main(dry_run: bool = False, episodes: int = 1) -> None:
    """Entry point for RL training script."""

    env = init_env()

    # For dry-run we only initialise the environment
    observation, info = env.reset()
    logger.info("Environment initialised")

    if dry_run:
        env.close()
        logger.info("Dry run complete")
        return

    # Model and agent setup
    model = PolicyNetwork(env.observation_space, env.action_space)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    agent = RLAgent(env, model, optimizer)

    buffer = ReplayBuffer(capacity=500)
    batch_size = 32
    gamma = 0.99
    epsilon = 0.1

    for ep in range(episodes):
        obs, info = env.reset()
        mask = select_mask(env)

        if info.get("request_teampreview"):
            team_order = agent.choose_team(obs)
            obs, mask, _, done, info = env.step(team_order)
        else:
            done = False

        total_reward = 0.0
        while not done:
            action_idx = choose_action(agent, obs, mask, epsilon)
            next_obs, next_mask, reward, done, _ = env.step(action_idx)
            buffer.add(obs, action_idx, reward, next_obs, done)
            obs = next_obs
            mask = next_mask
            total_reward += reward

            if len(buffer) >= batch_size:
                (states, actions, rewards, next_states, dones) = buffer.sample(batch_size)
                states_t = torch.as_tensor(states, dtype=torch.float32)
                actions_t = torch.as_tensor(actions, dtype=torch.long)
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
                next_states_t = torch.as_tensor(next_states, dtype=torch.float32)
                dones_t = torch.as_tensor(dones, dtype=torch.float32)

                q_vals = model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = model(next_states_t).max(1)[0]
                targets = rewards_t + (1 - dones_t) * gamma * next_q
                loss = nn.functional.mse_loss(q_vals, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        logger.info("Episode %d reward=%.2f", ep + 1, total_reward)

    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="RL training script")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="initialise environment and exit",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="number of training episodes",
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run, episodes=args.episodes)
