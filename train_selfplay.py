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

# Import device utilities
from src.utils.device_utils import get_device, transfer_to_device, get_device_info
# Import optimizer utilities
from src.utils.optimizer_utils import save_training_state, load_training_state, create_scheduler

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


# Ensure bundled poke_env package is importable
POKE_ENV_DIR = ROOT_DIR / "copy_of_poke-env"
if str(POKE_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(POKE_ENV_DIR))

from src.env.pokemon_env import PokemonEnv  # noqa: E402
from src.state.state_observer import StateObserver  # noqa: E402
from src.action import action_helper  # noqa: E402
from src.agents import PolicyNetwork, ValueNetwork, RLAgent  # noqa: E402
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


def init_env(reward: str = "composite", reward_config: str | None = None, team_mode: str = "default", teams_dir: str | None = None, normalize_rewards: bool = True) -> PokemonEnv:
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
        val0 = rl_agent.get_value(obs0)

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
        probs0 = agent0.select_action(obs0, mask0)
        probs1 = agent1.select_action(obs1, mask1)
        rng = env.rng
        act0 = int(rng.choice(len(probs0), p=probs0))
        act1 = int(rng.choice(len(probs1), p=probs1))
        logp0 = float(np.log(probs0[act0] + 1e-8))
        
        # Get value through RLAgent interface to handle hidden states properly
        val0 = agent0.get_value(obs0)

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
    win_rate_threshold: float = 0.6,
    win_rate_window: int = 50,
    device: str = "auto",
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

    cfg = load_config(config_path)
    logger.info("Loading configuration from: %s", config_path)
    
    # Load all configuration from config file, with command line overrides
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
    
    # Win rate configuration from config file
    win_rate_threshold = float(cfg.get("win_rate_threshold", win_rate_threshold))
    win_rate_window = int(cfg.get("win_rate_window", win_rate_window))
    
    # Model management from config file
    load_model = cfg.get("load_model", load_model)
    if load_model is not None:
        load_model = str(load_model)
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

    writer = SummaryWriter() if tensorboard else None
    
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
    sample_env = init_env(reward=reward, reward_config=reward_config, team_mode=team_mode, teams_dir=teams_dir, normalize_rewards=True)
    
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
                reset_optimizer=args.reset_optimizer,
            )
            logger.info("Loaded training state from %s, starting from episode %d", load_model, start_episode)
        except Exception as e:
            logger.error("Failed to load model from %s: %s", load_model, e)
            raise

    
    # For tracking opponent usage
    opponent_stats = {}
    
    # Win rate based opponent update system
    recent_battle_results = []  # Store recent battle results (1=win, 0=draw, -1=loss)
    # Use parameters passed to function
    opponent_snapshots = {}  # Store opponent network snapshots
    current_opponent_id = 0  # Track which opponent snapshot is being used
    last_opponent_update_episode = -1  # Track when opponent was last updated

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

    for ep in range(start_episode, start_episode + episodes):
        start_time = time.perf_counter()
        
        # Prepare environments and agents for this episode
        envs = []
        agents = []
        episode_opponents = []
        
        for i in range(max(1, parallel)):
            # Create new environment for this episode
            env = init_env(reward=reward, reward_config=reward_config, team_mode=team_mode, teams_dir=teams_dir, normalize_rewards=True)
            
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
            
            if opp_type == "self":
                # Self-play with win rate based opponent updates
                
                # Check if we need to update opponent based on recent win rate
                if ep > 0 and should_update_opponent(ep + 1, recent_battle_results, win_rate_window, win_rate_threshold):
                    # Create new opponent snapshot
                    current_opponent_id += 1
                    opponent_snapshots[current_opponent_id] = create_opponent_snapshot(
                        policy_net, value_net, network_config, env, current_opponent_id
                    )
                    last_opponent_update_episode = ep
                    # Clear recent results to avoid immediate re-update
                    recent_battle_results = []
                
                # Use existing opponent snapshot or create initial one
                if current_opponent_id not in opponent_snapshots:
                    # Create initial opponent snapshot (first episode)
                    current_opponent_id = 1
                    opponent_snapshots[current_opponent_id] = create_opponent_snapshot(
                        policy_net, value_net, network_config, env, current_opponent_id
                    )
                    last_opponent_update_episode = ep
                
                # Get opponent networks from snapshot
                opponent_policy_net, opponent_value_net = opponent_snapshots[current_opponent_id]
                
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

        # Run episodes
        if opponent_pool or opponent:
            # Mixed or single opponent mode
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

        # Use a temporary agent for updating
        temp_env = init_env(reward=reward, reward_config=reward_config, team_mode=team_mode, teams_dir=teams_dir, normalize_rewards=True)
        temp_agent = RLAgent(temp_env, policy_net, value_net, optimizer, algorithm=algorithm)
        
        if algo_name == "ppo":
            for i in range(ppo_epochs):
                loss = temp_agent.update(combined)
                logger.info("Episode %d epoch %d loss %.4f", ep + 1, i + 1, loss)
                if writer:
                    writer.add_scalar("loss", loss, ep * ppo_epochs + i)
        else:
            loss = temp_agent.update(combined)
            logger.info("Episode %d loss %.4f", ep + 1, loss)
            if writer:
                writer.add_scalar("loss", loss, ep + 1)
        
        # Reset hidden states after training to ensure clean state for next episode
        temp_agent.reset_hidden_states()
        
        temp_env.close()

        total_reward = sum(reward_list)
        duration = time.perf_counter() - start_time
        
        # Calculate sub-reward totals
        sub_totals = {}
        for logs in sub_logs_list:
            for name, val in logs.items():
                sub_totals[name] = sub_totals.get(name, 0.0) + val
        
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
        
        if writer:
            writer.add_scalar("reward", total_reward, ep + 1)
            writer.add_scalar("time/episode", duration, ep + 1)
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("learning_rate", current_lr, ep + 1)
            for name, val in sub_totals.items():
                writer.add_scalar(f"sub_reward/{name}", val, ep + 1)
        
        # Step scheduler if present
        if scheduler is not None:
            scheduler.step()

        if checkpoint_interval and (ep + 1) % checkpoint_interval == 0:
            try:
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

    if writer:
        writer.close()
    if save_path is not None:
        path = Path(save_path)
        try:
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
        win_rate_threshold=args.win_rate_threshold,
        win_rate_window=args.win_rate_window,
        device=args.device,
    )
