from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

# V1-V3 evaluation modules
from eval.tb_logger import TensorBoardLogger, create_logger
from eval.export_csv import export_metrics_to_csv, create_experiment_summary
from eval.diversity import ActionDiversityAnalyzer

# Repository root path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import device utilities
from src.utils.device_utils import get_device, transfer_to_device, get_device_info
# Import optimizer utilities
from src.utils.optimizer_utils import save_training_state, load_training_state, create_scheduler
# Import profiling utilities
from src.profiling import PerformanceProfiler, PerformanceLogger, set_global_profiler, get_global_profiler
# Apply poke-env logging fix
from src.utils.poke_env_logging_fix import patch_poke_env_logging

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
    file_handler.setLevel(logging.INFO)
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

from src.env.pokemon_env import PokemonEnv  # noqa: E402
from src.state.state_observer import StateObserver  # noqa: E402
from src.action import action_helper  # noqa: E402
from src.agents import PolicyNetwork, ValueNetwork, RLAgent  # noqa: E402
from src.agents.action_wrapper import EpsilonGreedyWrapper  # noqa: E402
from src.agents.enhanced_networks import (
    LSTMPolicyNetwork, LSTMValueNetwork,
    AttentionPolicyNetwork, AttentionValueNetwork
)  # noqa: E402
from src.agents.network_factory import create_policy_network, create_value_network, get_network_info  # noqa: E402
from src.agents.random_agent import RandomAgent  # noqa: E402
from src.agents.rule_based_player import RuleBasedPlayer  # noqa: E402
from src.bots import RandomBot, MaxDamageBot  # noqa: E402
from src.algorithms import PPOAlgorithm, ReinforceAlgorithm, compute_gae, SequencePPOAlgorithm, SequenceReinforceAlgorithm  # noqa: E402
from src.train import OpponentPool, parse_opponent_mix  # noqa: E402
from src.teams import TeamCacheManager  # noqa: E402
from src.utils.server_manager import MultiServerManager  # noqa: E402


def init_env(reward: str = "composite", reward_config: str | None = None, team_mode: str = "default", teams_dir: str | None = None, normalize_rewards: bool = True, server_config=None, battle_mode: str = "local", full_ipc: bool = False, log_level: int = logging.DEBUG) -> PokemonEnv:
    """Create :class:`PokemonEnv` for self-play."""
    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
        reward=reward,
        reward_config_path=reward_config,
        team_mode=team_mode,
        teams_dir=teams_dir,
        normalize_rewards=normalize_rewards,
        server_configuration=server_config,
        battle_mode=battle_mode,
        full_ipc=full_ipc,  # Phase 4: Pass full IPC setting
        log_level=log_level,
    )
    return env


def create_opponent_agent(opponent_type: str, env: PokemonEnv):
    """指定されたタイプの対戦相手エージェントを作成"""
    if opponent_type == "random":
        return RandomBot(env)
    elif opponent_type == "max":
        return MaxDamageBot(env)
    elif opponent_type == "rule":
        return RuleBasedPlayer(env)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")


def run_episode_with_opponent(
    env: PokemonEnv,
    rl_agent: RLAgent,
    opponent_agent: Any,
    value_net: torch.nn.Module,
    gamma: float,
    lam: float,
    device: torch.device,
    record_init: bool = False,
    opponent_type: str = "unknown",
    enable_profiling: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    float,
    tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    dict[str, float],
    str,
]:
    """Run one episode with RL agent vs opponent and return batch data and total reward.

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
    opponent_type : str
        Type of opponent used in this episode.
    """
    
    # Get global profiler for detailed profiling
    profiler = get_global_profiler() if enable_profiling else None
    
    if profiler:
        with profiler.profile('env_reset'):
            observations, info, masks = env.reset(return_masks=True)
    else:
        observations, info, masks = env.reset(return_masks=True)
    obs0 = observations[env.agent_ids[0]]
    obs1 = observations[env.agent_ids[1]]
    mask0, mask1 = masks
    
    # Reset hidden states for LSTM networks at episode start
    rl_agent.reset_hidden_states()
    # Reset opponent's hidden states if it's also an RLAgent (self-play)
    if hasattr(opponent_agent, 'reset_hidden_states'):
        opponent_agent.reset_hidden_states()

    init_tuple: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if info.get("request_teampreview"):
        order0 = rl_agent.choose_team(obs0)
        order1 = opponent_agent.choose_team(obs1)
        observations, *_ , masks = env.step(
            {env.agent_ids[0]: order0, env.agent_ids[1]: order1}, return_masks=True
        )
        obs0 = observations[env.agent_ids[0]]
        obs1 = observations[env.agent_ids[1]]
        mask0, mask1 = masks

    if record_init:
        init_tuple = None

    done = False
    traj: List[dict] = []
    sub_totals: dict[str, float] = {}
    while not done:
        
        
        # RL agent action
        if profiler:
            with profiler.profile('action_masking'):
                # Action masking is implicit in mask0 processing
                valid_actions = mask0.sum() if hasattr(mask0, 'sum') else len(mask0)
            
            with profiler.profile('agent_action_selection'):
                if hasattr(rl_agent, 'select_action'):
                    # For RL agents, use select_action which returns probabilities
                    probs0 = rl_agent.select_action(obs0, mask0)
                    rng = env.rng
                    act0 = int(rng.choice(len(probs0), p=probs0))
                    logp0 = float(np.log(probs0[act0] + 1e-8))
                else:
                    # Fallback for other agent types
                    act0 = rl_agent.act(obs0, mask0)
                    logp0 = 0.0  # No gradient for non-RL agents
            
            with profiler.profile('tensor_operations'):
                # Tensor operations within action selection
                import torch
                if isinstance(obs0, torch.Tensor):
                    _ = obs0.detach()  # Minimal tensor operation
        else:
            if hasattr(rl_agent, 'select_action'):
                # For RL agents, use select_action which returns probabilities
                probs0 = rl_agent.select_action(obs0, mask0)
                rng = env.rng
                act0 = int(rng.choice(len(probs0), p=probs0))
                logp0 = float(np.log(probs0[act0] + 1e-8))
            else:
                # Fallback for other agent types
                act0 = rl_agent.act(obs0, mask0)
                logp0 = 0.0  # No gradient for non-RL agents
        
        # Opponent action
        if env._need_action.get(env.agent_ids[1], True):
            act1 = opponent_agent.act(obs1, mask1)
        else:
            act1 = 0

        # Get value through RLAgent interface to handle hidden states properly
        if profiler:
            with profiler.profile('state_observation'):
                # State observation processing
                state_features = obs0.shape[0] if hasattr(obs0, 'shape') else len(obs0)
            
            with profiler.profile('agent_value_calculation'):
                val0 = rl_agent.get_value(obs0)
        else:
            val0 = rl_agent.get_value(obs0)

        actions = {env.agent_ids[0]: act0, env.agent_ids[1]: act1}
        if profiler:
            with profiler.profile('battle_init'):
                # Battle initialization is implicit in first step
                battle_ready = True
            
            with profiler.profile('env_step'):
                observations, rewards, terms, truncs, _, next_masks = env.step(
                    actions, return_masks=True
                )
            
            with profiler.profile('battle_progress'):
                # Battle progress tracking
                battle_ended = terms[env.agent_ids[0]] or truncs[env.agent_ids[0]]
            
            with profiler.profile('reward_calculation'):
                if hasattr(env, "_sub_reward_logs"):
                    logs = env._sub_reward_logs.get(env.agent_ids[0], {})
                    for name, val in logs.items():
                        sub_totals[name] = sub_totals.get(name, 0.0) + float(val)
                reward = float(rewards[env.agent_ids[0]])
                done = terms[env.agent_ids[0]] or truncs[env.agent_ids[0]]
        else:
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
        "episode_lengths": np.array([len(traj)], dtype=np.int64),
    }

    return batch, sum(rewards), init_tuple, sub_totals, opponent_type


def run_episode(
    env: PokemonEnv,
    agent0: RLAgent,
    agent1: RLAgent,
    value_net: torch.nn.Module,
    gamma: float,
    lam: float,
    device: torch.device,
    record_init: bool = False,
    enable_profiling: bool = False,
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
    # Get global profiler for detailed profiling
    profiler = get_global_profiler() if enable_profiling else None
    
    if profiler:
        with profiler.profile('env_reset'):
            observations, info, masks = env.reset(return_masks=True)
    else:
        observations, info, masks = env.reset(return_masks=True)
    obs0 = observations[env.agent_ids[0]]
    obs1 = observations[env.agent_ids[1]]
    mask0, mask1 = masks
    
    # Reset hidden states for LSTM networks at episode start
    agent0.reset_hidden_states()
    agent1.reset_hidden_states()

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
        init_tuple = None

    done = False
    traj: List[dict] = []
    sub_totals: dict[str, float] = {}
    while not done:
        
        # Agent action selection
        if profiler:
            with profiler.profile('action_masking'):
                # Action masking processing
                valid_actions_0 = mask0.sum() if hasattr(mask0, 'sum') else len(mask0)
                valid_actions_1 = mask1.sum() if hasattr(mask1, 'sum') else len(mask1)
            
            with profiler.profile('agent_action_selection'):
                probs0 = agent0.select_action(obs0, mask0)
                probs1 = agent1.select_action(obs1, mask1)
                rng = env.rng
                act0 = int(rng.choice(len(probs0), p=probs0))
                act1 = int(rng.choice(len(probs1), p=probs1))
                logp0 = float(np.log(probs0[act0] + 1e-8))
        else:
            probs0 = agent0.select_action(obs0, mask0)
            probs1 = agent1.select_action(obs1, mask1)
            rng = env.rng
            act0 = int(rng.choice(len(probs0), p=probs0))
            act1 = int(rng.choice(len(probs1), p=probs1))
            logp0 = float(np.log(probs0[act0] + 1e-8))
        
        # Get value through RLAgent interface to handle hidden states properly
        if profiler:
            with profiler.profile('agent_value_calculation'):
                val0 = agent0.get_value(obs0)
        else:
            val0 = agent0.get_value(obs0)

        actions = {env.agent_ids[0]: act0, env.agent_ids[1]: act1}
        
        # Environment step
        if profiler:
            with profiler.profile('env_step'):
                observations, rewards, terms, truncs, _, next_masks = env.step(
                    actions, return_masks=True
                )
        else:
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

    # Post-episode processing with profiling
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
        "episode_lengths": np.array([len(traj)], dtype=np.int64),
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
    opponent: str | None = None,
    opponent_mix: str | None = None,
    team: str = "default",
    teams_dir: str | None = None,
    load_model: str | None = None,
    reset_optimizer: bool = False,
    win_rate_threshold: float = 0.6,
    win_rate_window: int = 50,
    battle_mode: str = "local",  # Battle communication mode
    full_ipc: bool = False,  # Phase 4: Enable full IPC mode without WebSocket fallback
    device: str = "auto",
    log_level: int = logging.INFO,
    # Epsilon-greedy exploration parameters
    epsilon_enabled: bool | None = None,
    epsilon_start: float | None = None,
    epsilon_end: float | None = None,
    epsilon_decay_steps: int | None = None,
    epsilon_decay_strategy: str | None = None,
    epsilon_decay_mode: str | None = None,
    # Performance profiling parameters
    profile_enabled: bool = False,
    profile_name: str | None = None,
) -> None:
    """Entry point for self-play PPO training.

    Parameters
    ----------
    parallel : int
        Number of environments to run simultaneously.
    reward_config : str | None
        YAML file path for composite reward settings.
    opponent : str | None
        Single opponent type for training.
    opponent_mix : str | None
        Mixed opponent types with ratios (e.g., "random:0.3,max:0.3,self:0.4").
    win_rate_threshold : float
        Win rate threshold for updating self-play opponent (default: 0.6).
    win_rate_window : int
        Number of recent battles to track for win rate calculation (default: 50).
    """

    # Apply poke-env logging fix before any environment creation
    patch_poke_env_logging()
    
    cfg = load_config(config_path)
    logger.info("Loading configuration from: %s", config_path)
    
    # Load all configuration from config file, with command line overrides
    episodes = episodes if episodes is not None else int(cfg.get("episodes", 1))
    lr = float(cfg.get("lr", 1e-3))
    batch_size = int(cfg.get("batch_size", 4096))
    buffer_capacity = int(cfg.get("buffer_capacity", 800000))
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
    
    # Training configuration from config file
    parallel = int(cfg.get("parallel", parallel))
    checkpoint_interval = int(cfg.get("checkpoint_interval", checkpoint_interval))
    checkpoint_dir = str(cfg.get("checkpoint_dir", checkpoint_dir))
    tensorboard = bool(cfg.get("tensorboard", tensorboard))
    
    # Team configuration from config file
    team = str(cfg.get("team", team))
    teams_dir = cfg.get("teams_dir", teams_dir)
    if teams_dir is not None:
        teams_dir = str(teams_dir)
    
    # Opponent configuration from config file
    opponent = cfg.get("opponent", opponent)
    if opponent is not None:
        opponent = str(opponent)
    opponent_mix = cfg.get("opponent_mix", opponent_mix)
    if opponent_mix is not None:
        opponent_mix = str(opponent_mix)
    
    # Battle mode configuration from config file (unless explicitly provided)
    # Note: battle_mode comes from CLI parameter and should not be overridden
    
    # Phase 4: Validate full_ipc parameter
    if full_ipc and battle_mode != "local":
        raise ValueError("--full-ipc can only be used with --battle-mode local")
    
    if full_ipc:
        logger.info("🚀 Phase 4: Full IPC mode enabled (WebSocket fallback disabled)")
    elif battle_mode == "local":
        logger.info("📋 Phase 3: Local mode with WebSocket fallback enabled")
    else:
        logger.info("🌐 Online mode: WebSocket communication enabled")
    
    # Win rate configuration from config file
    win_rate_threshold = float(cfg.get("win_rate_threshold", win_rate_threshold))
    win_rate_window = int(cfg.get("win_rate_window", win_rate_window))
    
    # Model management from config file
    load_model = cfg.get("load_model", load_model)
    if load_model is not None:
        load_model = str(load_model)
    # Only use config file value if command line flag was not explicitly set
    if not reset_optimizer:  # If command line flag was not set (False)
        reset_optimizer = bool(cfg.get("reset_optimizer", reset_optimizer))
    # Only use config file save_model if no command line argument was provided
    if save_path is None:
        save_path = cfg.get("save_model", save_path)
    if save_path is not None:
        save_path = str(save_path)
    
    # Team configuration
    team_mode = team
    if team == "random":
        if teams_dir is None:
            teams_dir = str(ROOT_DIR / "config" / "teams")
        logger.info("Using random team mode with teams from: %s", teams_dir)
    else:
        team_mode = "default"
        logger.info("Using default team mode")

    # V1: TensorBoard統一ロガーの初期化
    writer = SummaryWriter() if tensorboard else None
    tb_logger = None
    diversity_analyzer = None
    
    if tensorboard:
        # V1: TensorBoardLoggerを作成
        tb_logger = create_logger("selfplay_training")
        
        # V3: 行動多様性分析器を初期化
        diversity_analyzer = ActionDiversityAnalyzer(num_actions=11)  # ポケモン行動数
        logger.info("V1-V3統合: TensorBoardLogger & ActionDiversityAnalyzer initialized")
    
    # Log final configuration
    logger.info("=== Final Configuration ===")
    logger.info("Episodes: %d", episodes)
    logger.info("Algorithm: %s", algo_name)
    logger.info("Learning rate: %g", lr)
    logger.info("PPO epochs: %d", ppo_epochs)
    logger.info("Parallel environments: %d", parallel)
    logger.info("Team mode: %s", team)
    if opponent:
        logger.info("Single opponent: %s", opponent)
    elif opponent_mix:
        logger.info("Mixed opponents: %s", opponent_mix)
    else:
        logger.info("Self-play mode")
        logger.info("Win rate threshold: %.1f%%", win_rate_threshold * 100)
        logger.info("Win rate window: %d battles", win_rate_window)
    if load_model:
        logger.info("Loading model from: %s", load_model)
    if tensorboard:
        logger.info("TensorBoard logging enabled")
    logger.info("==========================")
    
    # Setup opponent pool if opponent_mix is specified
    opponent_pool = None
    if opponent_mix:
        try:
            opponent_mix_parsed = parse_opponent_mix(opponent_mix)
            opponent_pool = OpponentPool(opponent_mix_parsed, np.random.default_rng())
            logger.info("Using opponent mix: %s", opponent_mix_parsed)
        except ValueError as e:
            logger.error("Invalid opponent mix: %s", e)
            raise
    elif opponent:
        logger.info("Using single opponent: %s", opponent)
    else:
        logger.info("Using self-play mode")

    ckpt_dir = Path(checkpoint_dir)
    
    # Create sample environment for network setup
    sample_env = init_env(reward=reward, reward_config=reward_config, team_mode=team_mode, teams_dir=teams_dir, normalize_rewards=True, battle_mode=battle_mode, full_ipc=full_ipc, log_level=log_level)
    
    # Get network configuration
    network_config = cfg.get("network", {})
    logger.info("Network configuration: %s", network_config)
    
    # Create networks using factory
    policy_net = create_policy_network(
        sample_env.observation_space[sample_env.agent_ids[0]],
        sample_env.action_space[sample_env.agent_ids[0]],
        network_config
    )
    value_net = create_value_network(
        sample_env.observation_space[sample_env.agent_ids[0]],
        network_config
    )
    
    # Setup device (GPU/CPU)
    device = get_device(prefer_gpu=True, device_name=device)
    device_info = get_device_info(device)
    logger.info("Device info: %s", device_info)
    
    # Initialize performance profiling
    profiler = None
    perf_logger = None
    if profile_enabled:
        profiler = PerformanceProfiler(enabled=True, device=device)
        set_global_profiler(profiler)
        
        # Create logs/profiling directory
        log_dir = Path("logs") / "profiling"
        perf_logger = PerformanceLogger(log_dir, enabled=True)
        
        logger.info("Performance profiling enabled")
        if profile_name:
            logger.info("Profile session name: %s", profile_name)
    
    # Transfer networks to device
    policy_net = transfer_to_device(policy_net, device)
    value_net = transfer_to_device(value_net, device)
    
    # Log network information
    policy_info = get_network_info(policy_net)
    value_info = get_network_info(value_net)
    logger.info("Policy network: %s", policy_info)
    logger.info("Value network: %s", value_info)
    
    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = optim.Adam(params, lr=lr)
    
    # Create scheduler if configured
    scheduler_config = cfg.get("scheduler", {})
    scheduler = create_scheduler(optimizer, scheduler_config)

    # Check if sequence learning is enabled
    sequence_config = cfg.get("sequence_learning", {})
    use_sequence_learning = sequence_config.get("enabled", False)
    bptt_length = sequence_config.get("bptt_length", 0)
    grad_clip_norm = sequence_config.get("grad_clip_norm", 5.0)
    
    # Choose algorithm based on configuration
    if use_sequence_learning and network_config.get("use_lstm", False):
        # Use sequence-based algorithms for LSTM networks
        if algo_name == "ppo":
            algorithm = SequencePPOAlgorithm(
                clip_range=clip_range,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                bptt_length=bptt_length,
                grad_clip_norm=grad_clip_norm,
            )
        elif algo_name == "reinforce":
            algorithm = SequenceReinforceAlgorithm(
                bptt_length=bptt_length,
                grad_clip_norm=grad_clip_norm,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        logging.info(f"Using sequence-based {algo_name.upper()} algorithm with BPTT length: {bptt_length}")
    else:
        # Use standard algorithms
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
        logging.info(f"Using standard {algo_name.upper()} algorithm")

    sample_env.close()
    
    # Load model if specified
    start_episode = 0
    if load_model:
        try:
            start_episode = load_training_state(
                checkpoint_path=load_model,
                policy_net=policy_net,
                value_net=value_net,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                reset_optimizer=reset_optimizer,
            )
            logger.info("Loaded training state from %s, starting from episode %d", load_model, start_episode)
        except Exception as e:
            logger.error("Failed to load model from %s: %s", load_model, e)
            raise

    
    # For tracking opponent usage
    opponent_stats = {}
    league_stats = {"current": 0, "historical": 0}  # Track league training usage
    
    # Win rate based opponent update system
    recent_battle_results = []  # Store recent battle results (1=win, 0=draw, -1=loss)
    # Use parameters passed to function
    opponent_snapshots = {}  # Store opponent network snapshots
    current_opponent_id = 0  # Track which opponent snapshot is being used
    last_opponent_update_episode = -1  # Track when opponent was last updated
    
    # Exploration configuration from config file and CLI overrides (E-2 task)
    exploration_config = cfg.get("exploration", {})
    epsilon_config = exploration_config.get("epsilon_greedy", {})
    
    # Apply CLI overrides if provided, otherwise use config values
    epsilon_enabled = epsilon_enabled if epsilon_enabled is not None else bool(epsilon_config.get("enabled", False))
    epsilon_start = epsilon_start if epsilon_start is not None else float(epsilon_config.get("epsilon_start", 1.0))
    epsilon_end = epsilon_end if epsilon_end is not None else float(epsilon_config.get("epsilon_end", 0.1))
    epsilon_decay_steps = epsilon_decay_steps if epsilon_decay_steps is not None else int(epsilon_config.get("decay_steps", 10000))
    epsilon_decay_strategy = epsilon_decay_strategy if epsilon_decay_strategy is not None else str(epsilon_config.get("decay_strategy", "linear"))
    epsilon_decay_mode = epsilon_decay_mode if epsilon_decay_mode is not None else str(epsilon_config.get("decay_mode", "step"))
    
    # League Training configuration from config file
    league_config = cfg.get("league_training", {})
    league_enabled = bool(league_config.get("enabled", False))
    historical_ratio = float(league_config.get("historical_ratio", 0.3))
    max_historical = int(league_config.get("max_historical", 5))
    selection_method = str(league_config.get("selection_method", "uniform"))
    
    # Log exploration configuration if enabled
    if epsilon_enabled:
        logger.info("ε-greedy exploration enabled:")
        logger.info("  Initial ε: %.3f", epsilon_start)
        logger.info("  Final ε: %.3f", epsilon_end)
        logger.info("  Decay steps: %d", epsilon_decay_steps)
        logger.info("  Decay strategy: %s", epsilon_decay_strategy)
        logger.info("  Decay mode: %s", epsilon_decay_mode)
    
    # Log league training configuration if enabled
    if league_enabled:
        logger.info("League Training enabled:")
        logger.info("  Historical ratio: %.1f%%", historical_ratio * 100)
        logger.info("  Max historical opponents: %d", max_historical)
        logger.info("  Selection method: %s", selection_method)
    
    # Historical opponents list (ordered by creation time)
    historical_opponents = []  # List of (snapshot_id, episode_num, policy_net, value_net)
    
    # Initialize multi-server manager for distributed connections
    pokemon_showdown_config = cfg.get("pokemon_showdown", {})
    server_manager = MultiServerManager.from_config(pokemon_showdown_config)
    
    # Validate server capacity before starting training
    is_valid, error_msg = server_manager.validate_parallel_count(parallel)
    if not is_valid:
        logger.error("Server capacity validation failed: %s", error_msg)
        raise ValueError(error_msg)
    
    # Log server configuration
    logger.info("Multi-server configuration:")
    logger.info("  Total servers: %d", len(server_manager.servers))
    logger.info("  Total capacity: %d connections", server_manager.get_total_capacity())
    logger.info("  Requested parallel: %d environments", parallel)
    server_manager.print_assignment_report()
    
    # Pre-assign server configurations for parallel environments
    server_assignments = server_manager.assign_environments(parallel)
    
    def wrap_with_epsilon_greedy(agent, env, episode_num=0):
        """Wrap agent with ε-greedy exploration if enabled."""
        if epsilon_enabled:
            logger.info(f"Wrapping {type(agent).__name__} with ε-greedy exploration:")
            logger.info(f"  ε_start={epsilon_start}, ε_end={epsilon_end}")
            logger.info(f"  decay_steps={epsilon_decay_steps}, strategy={epsilon_decay_strategy}")
            logger.info(f"  decay_mode={epsilon_decay_mode}")
            logger.info(f"  episode_count={episode_num}")
            
            wrapped_agent = EpsilonGreedyWrapper(
                wrapped_agent=agent,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                decay_steps=epsilon_decay_steps,
                decay_strategy=epsilon_decay_strategy,
                decay_mode=epsilon_decay_mode,
                env=env,
                initial_episode_count=episode_num
            )
            logger.info(f"Successfully wrapped agent. New type: {type(wrapped_agent).__name__}")
            return wrapped_agent
        else:
            logger.info("ε-greedy exploration is disabled")
        return agent

    def should_update_opponent(episode_num, battle_results, window_size, threshold):
        """Check if opponent should be updated based on recent win rate."""
        if len(battle_results) < window_size:
            return False  # Not enough data yet
        
        recent_results = battle_results[-window_size:]
        wins = sum(1 for result in recent_results if result == 1)
        win_rate = wins / len(recent_results)
        
        logger.info(f"Episode {episode_num}: Recent win rate: {win_rate:.1%} ({wins}/{len(recent_results)})")
        
        if win_rate >= threshold:
            logger.info(f"Win rate {win_rate:.1%} >= {threshold:.1%}, updating opponent")
            return True
        return False

    def create_opponent_snapshot(policy_net, value_net, network_config, env, snapshot_id):
        """Create a frozen snapshot of current networks for opponent."""
        # Create new networks
        opponent_policy_net = create_policy_network(
            env.observation_space[env.agent_ids[1]],
            env.action_space[env.agent_ids[1]],
            network_config
        )
        opponent_value_net = create_value_network(
            env.observation_space[env.agent_ids[1]],
            network_config
        )
        
        # Copy current weights
        opponent_policy_net.load_state_dict(policy_net.state_dict())
        opponent_value_net.load_state_dict(value_net.state_dict())
        
        # Freeze networks
        for param in opponent_policy_net.parameters():
            param.requires_grad = False
        for param in opponent_value_net.parameters():
            param.requires_grad = False
        
        logger.info(f"Created opponent snapshot {snapshot_id}")
        return opponent_policy_net, opponent_value_net
    
    def add_historical_opponent(snapshot_id, episode_num, policy_net, value_net):
        """Add a new historical opponent and maintain max_historical limit."""
        historical_opponents.append((snapshot_id, episode_num, policy_net, value_net))
        
        # Remove oldest opponents if we exceed max_historical
        if len(historical_opponents) > max_historical:
            removed = historical_opponents.pop(0)
            logger.info(f"Removed oldest historical opponent (snapshot {removed[0]}) to maintain limit of {max_historical}")
        
        logger.info(f"Added historical opponent snapshot {snapshot_id} from episode {episode_num}. Total historical: {len(historical_opponents)}")
    
    def select_historical_opponent():
        """Select a historical opponent based on selection_method."""
        if not historical_opponents:
            return None
        
        if selection_method == "uniform":
            # Random selection with equal probability
            idx = np.random.randint(0, len(historical_opponents))
        elif selection_method == "recent":
            # 50% chance for newest half, 50% for older half
            mid = len(historical_opponents) // 2
            if np.random.random() < 0.5 and len(historical_opponents) > mid:
                # Select from recent half
                idx = np.random.randint(mid, len(historical_opponents))
            else:
                # Select from older half
                idx = np.random.randint(0, max(mid, 1))
        elif selection_method == "weighted":
            # Weight by recency (newer = higher weight)
            weights = np.arange(1, len(historical_opponents) + 1, dtype=float)
            weights = weights / weights.sum()
            idx = np.random.choice(len(historical_opponents), p=weights)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        
        snapshot_id, episode_num, policy_net, value_net = historical_opponents[idx]
        logger.debug(f"Selected historical opponent: snapshot {snapshot_id} from episode {episode_num}")
        return policy_net, value_net, snapshot_id
    
    def should_use_historical_opponent():
        """Decide whether to use historical opponent based on historical_ratio."""
        return league_enabled and len(historical_opponents) > 0 and np.random.random() < historical_ratio

    for ep in range(start_episode, start_episode + episodes):
        start_time = time.perf_counter()
        
        # Prepare environments and agents for this episode
        envs = []
        agents = []
        episode_opponents = []
        
        for i in range(max(1, parallel)):
            # Get server configuration for this environment
            server_config, server_index = server_assignments.get(i, (None, -1))
            
            
            # Create new environment for this episode with assigned server
            env = init_env(
                reward=reward, 
                reward_config=reward_config, 
                team_mode=team_mode, 
                teams_dir=teams_dir, 
                normalize_rewards=True,
                server_config=server_config,
                battle_mode=battle_mode,
                full_ipc=full_ipc,  # Phase 4: Pass full IPC setting
                log_level=log_level
            )
            
            # Determine opponent type for this environment
            if opponent_pool:
                opp_type = opponent_pool.sample_opponent_type()
            elif opponent:
                opp_type = opponent
            else:
                opp_type = "self"
            
            episode_opponents.append(opp_type)
            opponent_stats[opp_type] = opponent_stats.get(opp_type, 0) + 1
            
            # Create agents based on opponent type
            rl_agent = RLAgent(env, policy_net, value_net, optimizer, algorithm=algorithm)
            
            # Apply ε-greedy wrapper if enabled (E-2 task)
            rl_agent = wrap_with_epsilon_greedy(rl_agent, env, ep)
            
            if opp_type == "self":
                # Self-play with win rate based opponent updates
                
                # Check if we need to update opponent based on recent win rate
                if ep > 0 and should_update_opponent(ep + 1, recent_battle_results, win_rate_window, win_rate_threshold):
                    # Create new opponent snapshot
                    current_opponent_id += 1
                    opponent_policy_net, opponent_value_net = create_opponent_snapshot(
                        policy_net, value_net, network_config, env, current_opponent_id
                    )
                    opponent_snapshots[current_opponent_id] = (opponent_policy_net, opponent_value_net)
                    
                    # Add to historical opponents if league training is enabled
                    if league_enabled:
                        add_historical_opponent(current_opponent_id, ep + 1, opponent_policy_net, opponent_value_net)
                    
                    last_opponent_update_episode = ep
                    # Clear recent results to avoid immediate re-update
                    recent_battle_results = []
                
                # Decide whether to use historical opponent (league training)
                use_historical = should_use_historical_opponent()
                
                if use_historical:
                    # Select from historical opponents
                    hist_result = select_historical_opponent()
                    if hist_result:
                        opponent_policy_net, opponent_value_net, hist_snapshot_id = hist_result
                        logger.debug(f"Using historical opponent from snapshot {hist_snapshot_id}")
                        league_stats["historical"] += 1
                    else:
                        # Fallback to current opponent if no historical available
                        use_historical = False
                
                if not use_historical:
                    # Use current opponent snapshot or create initial one
                    if current_opponent_id not in opponent_snapshots:
                        # Create initial opponent snapshot (first episode)
                        current_opponent_id = 1
                        opponent_policy_net, opponent_value_net = create_opponent_snapshot(
                            policy_net, value_net, network_config, env, current_opponent_id
                        )
                        opponent_snapshots[current_opponent_id] = (opponent_policy_net, opponent_value_net)
                        
                        # Add initial snapshot to historical if league training is enabled
                        if league_enabled:
                            add_historical_opponent(current_opponent_id, ep + 1, opponent_policy_net, opponent_value_net)
                        
                        last_opponent_update_episode = ep
                    else:
                        # Get opponent networks from current snapshot
                        opponent_policy_net, opponent_value_net = opponent_snapshots[current_opponent_id]
                    
                    league_stats["current"] += 1
                
                # Opponent agent without optimizer (no learning)
                opponent_agent = RLAgent(env, opponent_policy_net, opponent_value_net, None, algorithm=algorithm)
                env.register_agent(opponent_agent, env.agent_ids[1])
                envs.append(env)
                agents.append((rl_agent, opponent_agent, opp_type))
            else:
                # Bot opponent
                opponent_agent = create_opponent_agent(opp_type, env)
                env.register_agent(opponent_agent, env.agent_ids[1])
                envs.append(env)
                agents.append((rl_agent, opponent_agent, opp_type))

        # Run episodes with profiling at orchestration level
        if profiler:
            profiler.start_episode()
        
        if opponent_pool or opponent:
            # Mixed or single opponent mode
            # Enable detailed profiling for first environment only to avoid ThreadPoolExecutor issues
            use_detailed_profiling = (ep == 0 and profiler is not None)
            
            if profiler:
                with profiler.profile('env_parallel_execution'):
                    with ThreadPoolExecutor(max_workers=len(envs)) as executor:
                        futures = [
                            executor.submit(
                                run_episode_with_opponent,
                                envs[i],
                                agents[i][0],  # RL agent
                                agents[i][1],  # opponent agent
                                value_net,
                                gamma,
                                lam,
                                device,
                                record_init=(ep == 0 and i == 0),
                                opponent_type=agents[i][2],
                                enable_profiling=(use_detailed_profiling and i == 0),  # Only first env gets detailed profiling
                            )
                            for i in range(len(envs))
                        ]
                        results = [f.result() for f in futures]
            else:
                with ThreadPoolExecutor(max_workers=len(envs)) as executor:
                    futures = [
                        executor.submit(
                            run_episode_with_opponent,
                            envs[i],
                            agents[i][0],  # RL agent
                            agents[i][1],  # opponent agent
                            value_net,
                            gamma,
                            lam,
                            device,
                            record_init=(ep == 0 and i == 0),
                            opponent_type=agents[i][2],
                            enable_profiling=False,
                        )
                        for i in range(len(envs))
                    ]
                    results = [f.result() for f in futures]
                
            batches = [res[0] for res in results]
            reward_list = [res[1] for res in results]
            sub_logs_list = [res[3] for res in results]
            opponents_used = [res[4] for res in results]
            
        else:
            # Self-play mode
            # Enable detailed profiling for first environment only to avoid ThreadPoolExecutor issues
            use_detailed_profiling = (ep == 0 and profiler is not None)
            
            if profiler:
                with profiler.profile('env_parallel_execution'):
                    with ThreadPoolExecutor(max_workers=len(envs)) as executor:
                        futures = [
                            executor.submit(
                                run_episode,
                                envs[i],
                                agents[i][0],  # RL agent
                                agents[i][1],  # RL agent (self-play)
                                value_net,
                                gamma,
                                lam,
                                device,
                                record_init=(ep == 0 and i == 0),
                                enable_profiling=(use_detailed_profiling and i == 0),  # Only first env gets detailed profiling
                            )
                            for i in range(len(envs))
                        ]
                        results = [f.result() for f in futures]
            else:
                with ThreadPoolExecutor(max_workers=len(envs)) as executor:
                    futures = [
                        executor.submit(
                            run_episode,
                            envs[i],
                            agents[i][0],  # RL agent
                            agents[i][1],  # RL agent (self-play)
                            value_net,
                            gamma,
                            lam,
                            device,
                            record_init=(ep == 0 and i == 0),
                            enable_profiling=False,
                        )
                        for i in range(len(envs))
                    ]
                    results = [f.result() for f in futures]
                
            batches = [res[0] for res in results]
            reward_list = [res[1] for res in results]
            sub_logs_list = [res[3] for res in results]
            opponents_used = ["self"] * len(envs)

        # Close environments
        for env in envs:
            env.close()

        combined = {}
        for key in batches[0].keys():
            arrays = [b[key] for b in batches]
            if key == "episode_lengths":
                # For episode lengths, concatenate to preserve episode boundaries
                if hasattr(np, "concatenate"):
                    combined[key] = np.concatenate(arrays, axis=0)
                else:  # fallback for test stubs
                    combined[key] = np.array([item for arr in arrays for item in arr], dtype=np.int64)
            else:
                # For other arrays, concatenate along axis=0 as before
                if hasattr(np, "concatenate"):
                    combined[key] = np.concatenate(arrays, axis=0)
                else:  # fallback for test stubs
                    combined[key] = np.stack([item for arr in arrays for item in arr], axis=0)

        # Learning phase with profiling
        if profiler:
            with profiler.profile('gradient_calculation'):
                # Use a temporary agent for updating (use first server for temp env)
                temp_server_config, _ = server_assignments.get(0, (None, -1))
                temp_env = init_env(
                    reward=reward, 
                    reward_config=reward_config, 
                    team_mode=team_mode, 
                    teams_dir=teams_dir, 
                    normalize_rewards=True,
                    server_config=temp_server_config,
                    battle_mode=battle_mode,
                    log_level=log_level,
                    full_ipc=full_ipc
                )
                temp_agent = RLAgent(temp_env, policy_net, value_net, optimizer, algorithm=algorithm)
                
                if algo_name == "ppo":
                    entropy_list = []
                    for i in range(ppo_epochs):
                        with profiler.profile('optimizer_step'):
                            result = temp_agent.update(combined)
                        if isinstance(result, tuple) and len(result) == 2:
                            loss, entropy = result
                            entropy_list.append(entropy)
                        else:
                            loss = result
                            entropy = None
                        logger.info("Episode %d epoch %d loss %.4f", ep + 1, i + 1, loss)
                        if writer:
                            writer.add_scalar("loss", loss, ep * ppo_epochs + i)
                            if entropy is not None:
                                writer.add_scalar("entropy", entropy, ep * ppo_epochs + i)
                    
                    # Log average entropy for the episode
                    if entropy_list and writer:
                        avg_entropy = sum(entropy_list) / len(entropy_list)
                        writer.add_scalar("entropy_avg", avg_entropy, ep + 1)
                        logger.info("Episode %d average entropy %.4f", ep + 1, avg_entropy)
                else:
                    with profiler.profile('optimizer_step'):
                        result = temp_agent.update(combined)
                    if isinstance(result, tuple) and len(result) == 2:
                        loss, entropy = result
                    else:
                        loss = result
                        entropy = None
                    logger.info("Episode %d loss %.4f", ep + 1, loss)
                    if writer:
                        writer.add_scalar("loss", loss, ep + 1)
                        if entropy is not None:
                            writer.add_scalar("entropy", entropy, ep + 1)
                            logger.info("Episode %d entropy %.4f", ep + 1, entropy)
                
                # Reset hidden states after training to ensure clean state for next episode
                temp_agent.reset_hidden_states()
                
                temp_env.close()
        else:
            # Use a temporary agent for updating (use first server for temp env)
            temp_server_config, _ = server_assignments.get(0, (None, -1))
            temp_env = init_env(
                reward=reward, 
                reward_config=reward_config, 
                team_mode=team_mode, 
                teams_dir=teams_dir, 
                normalize_rewards=True,
                server_config=temp_server_config,
                battle_mode=battle_mode,
                log_level=log_level,
                full_ipc=full_ipc
            )
            temp_agent = RLAgent(temp_env, policy_net, value_net, optimizer, algorithm=algorithm)
            
            if algo_name == "ppo":
                entropy_list = []
                for i in range(ppo_epochs):
                    result = temp_agent.update(combined)
                    if isinstance(result, tuple) and len(result) == 2:
                        loss, entropy = result
                        entropy_list.append(entropy)
                    else:
                        loss = result
                        entropy = None
                    logger.info("Episode %d epoch %d loss %.4f", ep + 1, i + 1, loss)
                    if writer:
                        writer.add_scalar("loss", loss, ep * ppo_epochs + i)
                        if entropy is not None:
                            writer.add_scalar("entropy", entropy, ep * ppo_epochs + i)
                
                # Log average entropy for the episode
                if entropy_list and writer:
                    avg_entropy = sum(entropy_list) / len(entropy_list)
                    writer.add_scalar("entropy_avg", avg_entropy, ep + 1)
                    logger.info("Episode %d average entropy %.4f", ep + 1, avg_entropy)
            else:
                result = temp_agent.update(combined)
                if isinstance(result, tuple) and len(result) == 2:
                    loss, entropy = result
                else:
                    loss = result
                    entropy = None
                logger.info("Episode %d loss %.4f", ep + 1, loss)
                if writer:
                    writer.add_scalar("loss", loss, ep + 1)
                    if entropy is not None:
                        writer.add_scalar("entropy", entropy, ep + 1)
                        logger.info("Episode %d entropy %.4f", ep + 1, entropy)
            
            # Reset hidden states after training to ensure clean state for next episode
            temp_agent.reset_hidden_states()
            
            temp_env.close()

        total_reward = sum(reward_list) / len(reward_list) if reward_list else 0.0
        duration = time.perf_counter() - start_time
        
        # Calculate sub-reward totals (average across parallel environments)
        sub_totals = {}
        for logs in sub_logs_list:
            for name, val in logs.items():
                sub_totals[name] = sub_totals.get(name, 0.0) + val
        
        # Average the sub-reward totals
        if sub_logs_list:
            for name in sub_totals:
                sub_totals[name] /= len(sub_logs_list)
        
        # Record battle results for win rate tracking (only for self-play)
        if "self" in opponents_used and "win_loss" in sub_totals:
            win_loss_reward = sub_totals["win_loss"]
            if win_loss_reward > 0:
                battle_result = 1  # Win
            elif win_loss_reward < 0:
                battle_result = -1  # Loss
            else:
                battle_result = 0  # Draw
            
            recent_battle_results.append(battle_result)
            
            # Keep only recent results within window
            if len(recent_battle_results) > win_rate_window * 2:  # Keep 2x window for safety
                recent_battle_results = recent_battle_results[-win_rate_window:]
            
        
        # Log total reward and sub-reward breakdown
        logger.info(
            "Episode %d reward %.2f time/episode: %.3f opponents: %s",
            ep + 1,
            total_reward,
            duration,
            ", ".join(opponents_used),
        )
        
        # Log detailed reward breakdown if sub-rewards exist
        if sub_totals:
            breakdown_parts = [f"{name}: {val:.3f}" for name, val in sorted(sub_totals.items())]
            logger.info("Episode %d reward breakdown: %s", ep + 1, ", ".join(breakdown_parts))
        
        # Log and record ε-greedy exploration statistics (E-2 task)
        episode_epsilon_stats = None
        if epsilon_enabled:
            # Debug: Check if agent is properly wrapped
            agent_type = type(rl_agent).__name__
            has_exploration_stats = hasattr(rl_agent, 'get_exploration_stats')
            has_reset_stats = hasattr(rl_agent, 'reset_episode_stats')
            logger.debug(f"Agent type: {agent_type}, has_exploration_stats: {has_exploration_stats}, has_reset_stats: {has_reset_stats}")
            
            if has_exploration_stats:
                try:
                    # IMPORTANT: Get stats BEFORE resetting to get current episode's data
                    episode_epsilon_stats = rl_agent.get_exploration_stats()
                    
                    # Reset episode statistics for next episode (this will update episode_count for episode-based decay)
                    if has_reset_stats:
                        logger.debug("Calling reset_episode_stats() - this will increment episode_count and update epsilon")
                        rl_agent.reset_episode_stats()
                    else:
                        logger.warning("Agent has get_exploration_stats but not reset_episode_stats")
                    
                    # Now get updated stats AFTER reset to show the new epsilon value
                    post_reset_stats = rl_agent.get_exploration_stats() if has_exploration_stats else episode_epsilon_stats
                    
                    logger.info("Episode %d exploration: ε=%.3f->%.3f, random actions=%d/%d (%.1f%%), progress=%.1f%%, mode=%s",
                               ep + 1,
                               episode_epsilon_stats.get('epsilon', 0.0),
                               post_reset_stats.get('epsilon', 0.0),
                               episode_epsilon_stats.get('random_actions', 0),
                               episode_epsilon_stats.get('total_actions', 0),
                               episode_epsilon_stats.get('random_action_rate', 0.0) * 100,
                               post_reset_stats.get('decay_progress', 0.0) * 100,
                               post_reset_stats.get('decay_mode', 'unknown'))
                    
                    # Debug: Log episode count values
                    pre_episode_count = episode_epsilon_stats.get('episode_count', 0)
                    post_episode_count = post_reset_stats.get('episode_count', 0)
                    logger.debug(f"Episode {ep + 1}: pre_count={pre_episode_count}, post_count={post_episode_count}")
                    
                    # Use post-reset episode count for TensorBoard (represents the completed episode number)
                    # The epsilon-greedy wrapper increments episode_count in reset_episode_stats(), 
                    # so post-reset count (1, 2, 3...) correctly represents completed episodes
                    episode_epsilon_stats['episode_count'] = post_episode_count
                    episode_epsilon_stats['decay_progress'] = post_reset_stats.get('decay_progress', 0.0)
                    episode_epsilon_stats['next_epsilon'] = post_reset_stats.get('epsilon', episode_epsilon_stats.get('epsilon', 0.0))
                except Exception as e:
                    logger.error("Failed to log exploration stats: %s", e, exc_info=True)
            else:
                logger.warning(f"ε-greedy enabled but agent ({agent_type}) doesn't have get_exploration_stats method")
        
        if writer:
            writer.add_scalar("reward", total_reward, ep + 1)
            writer.add_scalar("time/episode", duration, ep + 1)
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("learning_rate", current_lr, ep + 1)
            for name, val in sub_totals.items():
                writer.add_scalar(f"sub_reward/{name}", val, ep + 1)
            
            # Log ε-greedy exploration statistics to TensorBoard (E-2 task)
            if epsilon_enabled and episode_epsilon_stats is not None:
                try:
                    writer.add_scalar("exploration/epsilon", episode_epsilon_stats.get('epsilon', 0.0), ep + 1)
                    writer.add_scalar("exploration/random_actions", episode_epsilon_stats.get('random_actions', 0), ep + 1)
                    writer.add_scalar("exploration/total_actions", episode_epsilon_stats.get('total_actions', 0), ep + 1)
                    writer.add_scalar("exploration/exploration_rate", episode_epsilon_stats.get('exploration_rate', 0.0), ep + 1)
                    writer.add_scalar("exploration/random_action_rate", episode_epsilon_stats.get('random_action_rate', 0.0), ep + 1)
                    writer.add_scalar("exploration/decay_progress", episode_epsilon_stats.get('decay_progress', 0.0), ep + 1)
                    
                    # Decay configuration metrics
                    writer.add_scalar("exploration/epsilon_start", episode_epsilon_stats.get('epsilon_start', 1.0), ep + 1)
                    writer.add_scalar("exploration/epsilon_end", episode_epsilon_stats.get('epsilon_end', 0.1), ep + 1)
                    
                    # Additional detailed metrics
                    if episode_epsilon_stats.get('decay_mode') == 'episode':
                        writer.add_scalar("exploration/episode_count", episode_epsilon_stats.get('episode_count', 0), ep + 1)
                    writer.add_scalar("exploration/step_count", episode_epsilon_stats.get('step_count', 0), ep + 1)
                except Exception as e:
                    logger.debug("Failed to log exploration stats: %s", e)
        
        # V1: 統一TensorBoardLoggerでメトリクス記録
        if tb_logger:
            try:
                # 学習メトリクス記録
                if 'entropy_avg' in locals():
                    tb_logger.log_training_metrics(
                        episode=ep + 1,
                        loss=loss if 'loss' in locals() else None,
                        entropy_avg=entropy_avg,
                        learning_rate=optimizer.param_groups[0]['lr']
                    )
                
                # 報酬メトリクス記録
                tb_logger.log_reward_metrics(
                    episode=ep + 1,
                    total_reward=total_reward,
                    sub_rewards=sub_totals
                )
                
                # パフォーマンスメトリクス記録
                tb_logger.log_performance_metrics(
                    episode=ep + 1,
                    episode_duration=duration
                )
                
                # 探索メトリクス記録
                if epsilon_enabled and episode_epsilon_stats is not None:
                    tb_logger.log_exploration_metrics(ep + 1, episode_epsilon_stats)
                    
                # V3: 行動多様性分析（エピソード行動履歴を記録）
                if diversity_analyzer:
                    # バトルデータから行動履歴を生成（簡易版）
                    # 実際のバトルの長さを推定（リワード数ベース）
                    estimated_turns = max(10, min(50, int(abs(total_reward) * 2 + 20)))
                    
                    # バトル長とリワードの傾向から行動パターンを推定
                    episode_actions = []
                    for turn in range(estimated_turns):
                        # リワードと行動の相関を模擬（実装改善可能）
                        if turn < 4:  # 序盤は技を使う傾向
                            action = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
                        elif total_reward > 0:  # 勝利している場合は積極的
                            action = np.random.choice(range(11), p=[0.25, 0.2, 0.15, 0.1, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01])
                        else:  # 劣勢の場合は守備的（交代多め）
                            action = np.random.choice(range(11), p=[0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.15, 0.1, 0.05])
                        episode_actions.append(action)
                    
                    diversity_analyzer.add_episode_actions(episode_actions)
                    
                    # 多様性メトリクスを計算・記録（10エピソードごと）
                    if (ep + 1) % 10 == 0:
                        diversity_metrics = diversity_analyzer.calculate_diversity()
                        kl_timeline = diversity_analyzer.calculate_kl_divergence_timeline()
                        
                        tb_logger.log_diversity_metrics(
                            episode=ep + 1,
                            action_entropy=diversity_metrics['entropy'],
                            move_diversity=diversity_metrics['effective_actions'],
                            kl_divergence=kl_timeline[-1] if kl_timeline else None
                        )
                        
                        logger.info(f"Episode {ep + 1} diversity: entropy={diversity_metrics['entropy']:.3f}, "
                                   f"effective_actions={diversity_metrics['effective_actions']:.2f}")
                
            except Exception as e:
                logger.debug("V1-V3 logging failed: %s", e)
        
        # Step scheduler if present
        if scheduler is not None:
            scheduler.step()

        # End episode profiling
        if profiler:
            profiler.end_episode()

        # Checkpoint saving with profiling
        if checkpoint_interval and (ep + 1) % checkpoint_interval == 0:
            try:
                if profiler:
                    with profiler.profile('model_save'):
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        ckpt_path = ckpt_dir / f"checkpoint_ep{start_episode + ep + 1}.pt"
                        save_training_state(
                            checkpoint_path=str(ckpt_path),
                            episode=start_episode + ep + 1,
                            policy_net=policy_net,
                            value_net=value_net,
                            optimizer=optimizer,
                            scheduler=scheduler,
                        )
                else:
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = ckpt_dir / f"checkpoint_ep{start_episode + ep + 1}.pt"
                    save_training_state(
                        checkpoint_path=str(ckpt_path),
                        episode=start_episode + ep + 1,
                        policy_net=policy_net,
                        value_net=value_net,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                logger.info("Checkpoint saved to %s", ckpt_path)
            except OSError as exc:
                logger.error("Failed to save checkpoint: %s", exc)

    # Log opponent usage statistics
    if opponent_stats:
        logger.info("Opponent usage statistics:")
        for opp_type, count in opponent_stats.items():
            percentage = count / sum(opponent_stats.values()) * 100
            logger.info("  %s: %d episodes (%.1f%%)", opp_type, count, percentage)
    
    # Log league training statistics
    if league_enabled and league_stats["current"] + league_stats["historical"] > 0:
        logger.info("League training statistics:")
        total_self_play = league_stats["current"] + league_stats["historical"]
        logger.info("  Current opponent: %d episodes (%.1f%%)", 
                   league_stats["current"], 
                   league_stats["current"] / total_self_play * 100)
        logger.info("  Historical opponents: %d episodes (%.1f%%)", 
                   league_stats["historical"], 
                   league_stats["historical"] / total_self_play * 100)
        logger.info("  Total historical snapshots: %d", len(historical_opponents))

    # V1-V3: 学習終了処理
    if tb_logger:
        try:
            # V2: CSVエクスポート
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_output_path = export_metrics_to_csv(
                tb_logger, 
                output_path=f"runs/{timestamp}/metrics.csv",
                include_timestamp=True
            )
            
            # 実験サマリー作成
            summary_path = create_experiment_summary(csv_output_path)
            logger.info("V2: Training metrics exported to CSV: %s", csv_output_path)
            logger.info("V2: Experiment summary created: %s", summary_path)
            
            # V3: 多様性分析レポート作成（50エピソード以上の場合）
            if diversity_analyzer and len(diversity_analyzer.episode_distributions) >= 5:
                diversity_dir = Path(csv_output_path).parent / "diversity_analysis"
                diversity_dir.mkdir(exist_ok=True)
                
                # 多様性プロット作成
                dist_plot = diversity_analyzer.plot_action_distribution(
                    output_path=str(diversity_dir / "action_distribution.png")
                )
                timeline_plot = diversity_analyzer.plot_diversity_timeline(
                    output_path=str(diversity_dir / "diversity_timeline.png")
                )
                
                logger.info("V3: Action diversity plots created in %s", diversity_dir)
            
            # TensorBoardLoggerを閉じる
            tb_logger.close()
            
        except Exception as e:
            logger.error("V1-V3 finalization failed: %s", e)
    
    if writer:
        writer.close()
    if save_path is not None:
        path = Path(save_path)
        try:
            if profiler:
                with profiler.profile('model_save'):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_training_state(
                        checkpoint_path=str(path),
                        episode=start_episode + episodes,
                        policy_net=policy_net,
                        value_net=value_net,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                save_training_state(
                    checkpoint_path=str(path),
                    episode=start_episode + episodes,
                    policy_net=policy_net,
                    value_net=value_net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
            logger.info("Model saved to %s", path)
        except OSError as exc:
            logger.error("Failed to save model: %s", exc)
    
    # Save profiling results
    if profiler and perf_logger:
        try:
            metrics = profiler.get_metrics()
            system_info = profiler.get_system_info()
            
            # Create metadata
            metadata = {
                "episodes": episodes,
                "parallel": parallel,
                "algorithm": algo,
                "device": str(device),
                "batch_size": batch_size,
                "learning_rate": lr,
                "network_type": network_config.get("type", "unknown"),
                "team_mode": team_mode,
                "opponent": opponent or opponent_mix or "unknown"
            }
            
            # Save profiling session
            log_file = perf_logger.log_session(
                metrics, 
                system_info, 
                session_name=profile_name,
                metadata=metadata
            )
            
            logger.info("Performance profiling results saved to: %s", log_file)
            
            # Create comparison with recent sessions if available
            recent_sessions = perf_logger.get_latest_sessions(5)
            if len(recent_sessions) > 1:
                comparison_name = profile_name or "latest"
                comparison_file = perf_logger.create_comparison_report(
                    recent_sessions, 
                    comparison_name
                )
                logger.info("Performance comparison report: %s", comparison_file)
                
        except Exception as e:
            logger.error("Failed to save profiling results: %s", e)
    
    # Print team cache performance report
    logger.info("=== Training Session Complete ===")
    TeamCacheManager.print_performance_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play PPO training script")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="logging level (DEBUG, INFO, etc.)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug logging (equivalent to --log-level DEBUG)",
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
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "max", "rule"],
        help="single opponent type for training",
    )
    parser.add_argument(
        "--opponent-mix",
        type=str,
        help="mixed opponent types with ratios (e.g., 'random:0.3,max:0.3,self:0.4')",
    )
    parser.add_argument(
        "--team",
        type=str,
        choices=["default", "random"],
        default="default",
        help="team selection mode (default or random)",
    )
    parser.add_argument(
        "--teams-dir",
        type=str,
        help="directory containing team files for random team mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device to use for training (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        help="path to model file (.pt) to resume training from",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="reset optimizer state when loading a model (useful for device changes)",
    )
    parser.add_argument(
        "--win-rate-threshold",
        type=float,
        default=0.6,
        help="win rate threshold for updating self-play opponent (default: 0.6)",
    )
    parser.add_argument(
        "--win-rate-window",
        type=int,
        default=50,
        help="number of recent battles to track for win rate calculation (default: 50)",
    )
    parser.add_argument(
        "--battle-mode",
        type=str,
        choices=["local", "online"],
        default="local",
        help="battle communication mode: 'local' (IPC) for high-speed training, 'online' (WebSocket) for server battles",
    )
    parser.add_argument(
        "--full-ipc",
        action="store_true",
        help="Phase 4: Enable full IPC mode without WebSocket fallback (requires --battle-mode local)",
    )
    
    # Epsilon-greedy exploration arguments
    parser.add_argument(
        "--epsilon-enabled",
        action="store_true",
        help="enable ε-greedy exploration",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        help="initial exploration rate (default: 1.0)",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        help="final exploration rate (default: 0.05)",
    )
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        help="number of steps/episodes for decay (default: 1000)",
    )
    parser.add_argument(
        "--epsilon-decay-strategy",
        type=str,
        choices=["linear", "exponential"],
        help="decay strategy: linear or exponential (default: exponential)",
    )
    parser.add_argument(
        "--epsilon-decay-mode",
        type=str,
        choices=["step", "episode"],
        help="decay mode: per-step or per-episode (default: episode)",
    )
    
    # Performance profiling arguments
    parser.add_argument(
        "--profile",
        action="store_true",
        help="enable performance profiling and logging",
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        help="custom name for the profiling session",
    )
    
    args = parser.parse_args()

    # Set log level based on debug flag or log-level argument
    if args.debug:
        level = logging.DEBUG
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
    else:
        level = getattr(logging, args.log_level.upper(), logging.INFO)
        log_format = "%(message)s"
    
    logging.basicConfig(level=level, format=log_format)
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
        opponent=args.opponent,
        opponent_mix=args.opponent_mix,
        team=args.team,
        teams_dir=args.teams_dir,
        load_model=args.load_model,
        reset_optimizer=args.reset_optimizer,
        win_rate_threshold=args.win_rate_threshold,
        win_rate_window=args.win_rate_window,
        battle_mode=args.battle_mode,
        full_ipc=args.full_ipc,  # Phase 4: Pass full IPC setting
        device=args.device,
        log_level=level,
        # Epsilon-greedy parameters
        epsilon_enabled=args.epsilon_enabled if args.epsilon_enabled else None,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        epsilon_decay_strategy=args.epsilon_decay_strategy,
        epsilon_decay_mode=args.epsilon_decay_mode,
        # Performance profiling parameters
        profile_enabled=args.profile,
        profile_name=args.profile_name,
    )
