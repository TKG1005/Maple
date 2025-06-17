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
from src.agents import PolicyNetwork, RLAgent, ReplayBuffer  # noqa: E402
import torch  # noqa: E402
from torch import optim  # noqa: E402


def init_env() -> SingleAgentCompatibilityWrapper:
    """Create :class:`PokemonEnv` wrapped for single-agent use."""

    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
    )
    return SingleAgentCompatibilityWrapper(env)


def train_step(agent: RLAgent, batch: dict[str, torch.Tensor]) -> None:
    """Perform a single policy gradient step using REINFORCE."""

    obs = torch.as_tensor(batch["observations"], dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)

    logits = agent.model(obs)
    log_probs = torch.log_softmax(logits, dim=-1)
    selected = log_probs[torch.arange(len(actions)), actions]
    loss = -(selected * rewards).mean()

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()


def main(*, dry_run: bool = False, episodes: int = 1, save_path: str | None = None) -> None:
    """Entry point for RL training script."""

    env = init_env()

    if dry_run:
        # 初期化のみ確認して即終了
        env.reset()
        logger.info("Environment initialised")
        env.close()
        logger.info("Dry run complete")
        return

    observation_dim = env.observation_space.shape
    action_space = env.action_space

    model = PolicyNetwork(env.observation_space, action_space)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    agent = RLAgent(env, model, optimizer)
    buffer = ReplayBuffer(capacity=1000, observation_shape=observation_dim)
    batch_size = 32

    for ep in range(episodes):
        obs, info = env.reset()
        if info.get("request_teampreview"):
            team_cmd = agent.choose_team(obs)
            obs, action_mask, _, done, _ = env.step(team_cmd)
        else:
            battle = env.env._current_battles[env.env.agent_ids[0]]
            action_mask, _ = action_helper.get_available_actions_with_details(battle)
            done = False

        total_reward = 0.0
        while not done:
            action = agent.act(obs, action_mask)
            next_obs, action_mask, reward, done, _ = env.step(action)
            buffer.add(obs, action, float(reward), done, next_obs)
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                train_step(agent, batch)
            obs = next_obs
            total_reward += float(reward)

        logger.info("Episode %d reward %.2f", ep + 1, total_reward)

    env.close()

    if save_path is not None:
        path = Path(save_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
            logger.info("Model saved to %s", path)
        except OSError as exc:
            logger.error("Failed to save model: %s", exc)


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
    parser.add_argument(
        "--save",
        type=str,
        metavar="FILE",
        help="path to save trained model (.pt)",
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run, episodes=args.episodes, save_path=args.save)

