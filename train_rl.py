from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import time
import yaml
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


# Use poke_env from .venv instead of copy_of_poke-env

from src.env.wrappers import SingleAgentCompatibilityWrapper  # noqa: E402
from src.env.pokemon_env import PokemonEnv  # noqa: E402
from src.state.state_observer import StateObserver  # noqa: E402
from src.action import action_helper  # noqa: E402
from src.agents import PolicyNetwork, ValueNetwork, RLAgent, ReplayBuffer  # noqa: E402
from src.agents.random_agent import RandomAgent  # noqa: E402
from src.agents.rule_based_player import RuleBasedPlayer  # noqa: E402
from src.algorithms import (
    BaseAlgorithm,
    ReinforceAlgorithm,
    DummyAlgorithm,
)  # noqa: E402
import torch  # noqa: E402
from torch import optim  # noqa: E402


def create_opponent_agent(opponent_type: str, env: PokemonEnv):
    """指定されたタイプの対戦相手エージェントを作成"""
    if opponent_type == "random":
        return RandomAgent(env)
    elif opponent_type == "rule":
        return RuleBasedPlayer(env)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")


def init_env(opponent_type: str = "random"):
    """Create PokemonEnv with specified opponent type."""
    
    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
    )
    
    if opponent_type == "random":
        # 従来通りの単一エージェント環境（ランダム相手）
        return SingleAgentCompatibilityWrapper(env)
    else:
        # マルチエージェント環境で特定の対戦相手
        opponent_agent = create_opponent_agent(opponent_type, env)
        env.register_agent(opponent_agent, env.agent_ids[1])
        return env


def main(
    *,
    config_path: str = str(ROOT_DIR / "config" / "train_config.yml"),
    dry_run: bool = False,
    episodes: int | None = None,
    save_path: str | None = None,
    tensorboard: bool = False,
    checkpoint_interval: int = 0,
    checkpoint_dir: str = "checkpoints",
    algorithm: BaseAlgorithm | None = None,
    opponent: str = "random",
    clip: float | None = None,
    gae_lambda: float | None = None,
    ppo_epochs: int | None = None,
    value_coef: float | None = None,
) -> None:
    """Entry point for RL training script."""

    cfg = load_config(config_path)
    episodes = episodes if episodes is not None else int(cfg.get("episodes", 1))
    lr = float(cfg.get("lr", 1e-3))
    buffer_capacity = int(cfg.get("buffer_capacity", 1000))
    batch_size = int(cfg.get("batch_size", 32))
    clip = clip if clip is not None else float(cfg.get("clip", 0.0))
    gae_lambda = gae_lambda if gae_lambda is not None else float(cfg.get("gae_lambda", 1.0))
    ppo_epochs = ppo_epochs if ppo_epochs is not None else int(cfg.get("ppo_epochs", 1))
    value_coef = value_coef if value_coef is not None else float(cfg.get("value_coef", 0.0))

    writer = None
    global_step = 0

    env = init_env(opponent)

    if dry_run:
        # 初期化のみ確認して即終了
        env.reset(return_masks=True)
        logger.info("Environment initialised")
        env.close()
        if writer:
            writer.close()
        logger.info("Dry run complete")
        return

    if tensorboard:
        writer = SummaryWriter()

    # 環境タイプに応じて観測・行動空間を取得
    is_single_agent = isinstance(env, SingleAgentCompatibilityWrapper)
    
    if is_single_agent:
        observation_dim = env.observation_space.shape
        action_space = env.action_space
        obs_space = env.observation_space
    else:
        # マルチエージェント環境の場合、学習エージェント（player_0）の空間を使用
        player_id = env.agent_ids[0]
        observation_dim = env.observation_space[player_id].shape
        action_space = env.action_space[player_id]
        obs_space = env.observation_space[player_id]

    policy_net = PolicyNetwork(obs_space, action_space)
    value_net = ValueNetwork(obs_space)
    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = optim.Adam(params, lr=lr)
    algorithm = algorithm or ReinforceAlgorithm()
    agent = RLAgent(env, policy_net, value_net, optimizer, algorithm=algorithm)
    buffer = ReplayBuffer(capacity=buffer_capacity, observation_shape=observation_dim)

    ckpt_dir = Path(checkpoint_dir)

    for ep in range(episodes):
        start_time = time.perf_counter()
        
        if is_single_agent:
            # シングルエージェント環境（従来通り）
            obs, info, action_mask = env.reset(return_masks=True)
            if info.get("request_teampreview"):
                team_cmd = agent.choose_team(obs)
                obs, action_mask, _, done, _ = env.step(team_cmd, return_masks=True)
            else:
                done = False

            total_reward = 0.0
            while not done:
                action = agent.act(obs, action_mask)
                next_obs, action_mask, reward, done, _ = env.step(action, return_masks=True)
                buffer.add(obs, action, float(reward), done, next_obs)
                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    loss = agent.update(batch)
                    if writer:
                        writer.add_scalar("loss", loss, global_step)
                    global_step += 1
                obs = next_obs
                total_reward += float(reward)
        else:
            # マルチエージェント環境（対戦相手あり）
            observations, info, masks = env.reset(return_masks=True)
            player_id = env.agent_ids[0]
            opponent_id = env.agent_ids[1]
            
            obs = observations[player_id]
            action_mask = masks[0]  # player_0のマスク
            
            if info.get("request_teampreview"):
                team_cmd = agent.choose_team(obs)
                # 対戦相手も必要に応じてチーム選択（ここではランダム）
                opponent_agent = env._agents[opponent_id]
                opponent_team_cmd = opponent_agent.choose_team(observations[opponent_id])
                
                observations, *_, masks = env.step(
                    {player_id: team_cmd, opponent_id: opponent_team_cmd}, return_masks=True
                )
                obs = observations[player_id]
                action_mask = masks[0]

            done = False
            total_reward = 0.0
            
            while not done:
                # 学習エージェントの行動
                action = agent.act(obs, action_mask) if env._need_action[player_id] else 0
                
                # 対戦相手の行動
                opponent_agent = env._agents[opponent_id]
                opponent_obs = observations[opponent_id]
                opponent_mask = masks[1]
                opponent_action = opponent_agent.act(opponent_obs, opponent_mask) if env._need_action[opponent_id] else 0
                
                # 両方の行動を環境に送信
                observations, rewards, terms, truncs, _, masks = env.step(
                    {player_id: action, opponent_id: opponent_action}, return_masks=True
                )
                
                next_obs = observations[player_id]
                reward = rewards[player_id]
                done = terms[player_id] or truncs[player_id]
                action_mask = masks[0]
                
                # 学習エージェントのデータのみをバッファに追加
                buffer.add(obs, action, float(reward), done, next_obs)
                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    loss = agent.update(batch)
                    if writer:
                        writer.add_scalar("loss", loss, global_step)
                    global_step += 1
                
                obs = next_obs
                total_reward += float(reward)

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

        if checkpoint_interval and (ep + 1) % checkpoint_interval == 0:
            try:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"checkpoint_ep{ep + 1}.pt"
                torch.save(policy_net.state_dict(), ckpt_path)
                logger.info("Checkpoint saved to %s", ckpt_path)
            except OSError as exc:
                logger.error("Failed to save checkpoint: %s", exc)

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
    parser = argparse.ArgumentParser(description="RL training script")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "config" / "train_config.yml"),
        help="path to YAML config file",
    )
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
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="enable TensorBoard logging to the runs/ directory",
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
        "--clip",
        type=float,
        default=None,
        help="PPO clipping ratio",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=None,
        metavar="LAMBDA",
        help="GAE lambda parameter",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=None,
        help="number of PPO update epochs",
    )
    parser.add_argument(
        "--value-coef",
        type=float,
        default=None,
        help="coefficient for value loss",
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "rule"],
        default="random",
        help="opponent type for training (random or rule)",
    )
    args = parser.parse_args()

    setup_logging("logs", vars(args))

    main(
        config_path=args.config,
        dry_run=args.dry_run,
        episodes=args.episodes,
        save_path=args.save,
        tensorboard=args.tensorboard,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        clip=args.clip,
        gae_lambda=args.gae_lambda,
        ppo_epochs=args.ppo_epochs,
        value_coef=args.value_coef,
        opponent=args.opponent,
    )
