"""Multiprocess worker functions for distributed training."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

# Add project root to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Apply poke-env logging fix early
from src.utils.poke_env_logging_fix import patch_poke_env_logging
patch_poke_env_logging()

# Core imports
from src.env.pokemon_env import PokemonEnv
from src.state.state_observer import StateObserver
from src.action import action_helper
from src.agents import RLAgent
from src.agents.network_factory import create_policy_network, create_value_network
from src.agents.random_agent import RandomAgent
from src.agents.rule_based_player import RuleBasedPlayer
from src.bots import RandomBot, MaxDamageBot
from src.algorithms import PPOAlgorithm, ReinforceAlgorithm, compute_gae, SequencePPOAlgorithm, SequenceReinforceAlgorithm
from src.utils.device_utils import get_device, transfer_to_device

logger = logging.getLogger(__name__)


def init_env_in_process(
    reward: str = "composite",
    reward_config: str | None = None,
    team_mode: str = "default",
    teams_dir: str | None = None,
    normalize_rewards: bool = True,
    server_config: dict | None = None,
    log_level: int = logging.DEBUG
) -> PokemonEnv:
    """Create PokemonEnv in child process with independent POKE_LOOP."""
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
        log_level=log_level
    )
    return env


def create_agent_from_config(
    env: PokemonEnv,
    agent_config: dict,
    model_state_dict: dict | None = None,
    device_name: str = "cpu"
) -> Any:
    """Create agent from configuration in child process."""
    agent_type = agent_config.get("type", "rl")
    
    if agent_type == "rl":
        # Create RL agent with model state
        network_config = agent_config["network_config"]
        algorithm_config = agent_config["algorithm_config"]
        
        # Get the actual observation dimension from environment's state observer
        state_dim = env.state_observer.get_observation_dimension()
        
        # Create networks with correct observation space
        from gymnasium import spaces
        obs_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(state_dim,))
        action_space = spaces.Discrete(11)  # Standard Pokemon action space
        
        policy_net = create_policy_network(obs_space, action_space, network_config)
        value_net = create_value_network(obs_space, network_config)
        
        # Load model state if provided
        if model_state_dict:
            policy_net.load_state_dict(model_state_dict["policy"])
            value_net.load_state_dict(model_state_dict["value"])
        
        # Transfer to device
        device = get_device(prefer_gpu=False, device_name=device_name)
        policy_net = transfer_to_device(policy_net, device)
        value_net = transfer_to_device(value_net, device)
        
        # Create algorithm
        algo_name = algorithm_config.get("name", "ppo")
        if algo_name == "ppo":
            if algorithm_config.get("sequence_learning", False):
                algorithm = SequencePPOAlgorithm(
                    clip_range=algorithm_config.get("clip_range", 0.2),
                    bptt_length=algorithm_config.get("bptt_length", 0),
                    grad_clip_norm=algorithm_config.get("grad_clip_norm", 5.0)
                )
            else:
                algorithm = PPOAlgorithm(
                    clip_range=algorithm_config.get("clip_range", 0.2)
                )
        elif algo_name == "reinforce":
            if algorithm_config.get("sequence_learning", False):
                algorithm = SequenceReinforceAlgorithm(
                    bptt_length=algorithm_config.get("bptt_length", 0),
                    grad_clip_norm=algorithm_config.get("grad_clip_norm", 5.0)
                )
            else:
                algorithm = ReinforceAlgorithm()
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        
        # Create optimizer (None for frozen agents)
        optimizer = None
        if agent_config.get("learning", True):
            from torch import optim
            optimizer = optim.Adam(
                list(policy_net.parameters()) + list(value_net.parameters()),
                lr=algorithm_config.get("lr", 0.0003)
            )
        
        return RLAgent(env, policy_net, value_net, optimizer, algorithm=algorithm)
        
    elif agent_type == "random":
        return RandomAgent(env)
    elif agent_type == "rule":
        return RuleBasedPlayer(env)
    elif agent_type == "random_bot":
        return RandomBot(env)
    elif agent_type == "max_bot":
        return MaxDamageBot(env)
    elif agent_type == "max":  # Add support for "max" opponent type
        return MaxDamageBot(env)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_episode_process(
    env_config: dict,
    rl_agent_config: dict,
    opponent_agent_config: dict,
    model_state_dict: dict,
    gamma: float,
    lam: float,
    device_name: str,
    record_init: bool = False,
    episode_type: str = "opponent"
) -> tuple:
    """Run a single episode in a child process.
    
    Args:
        env_config: Environment configuration
        rl_agent_config: RL agent configuration
        opponent_agent_config: Opponent agent configuration  
        model_state_dict: Model parameters for RL agent
        gamma: Discount factor
        lam: GAE lambda
        device_name: Device name (cpu/cuda/mps)
        record_init: Whether to record initial state
        episode_type: "opponent" or "self_play"
        
    Returns:
        Tuple of (batch_data, total_reward, init_tuple, sub_totals, opponent_type)
    """
    try:
        # 1. Initialize environment in child process
        env = init_env_in_process(
            reward=env_config.get("reward", "composite"),
            reward_config=env_config.get("reward_config"),
            team_mode=env_config.get("team_mode", "default"),
            teams_dir=env_config.get("teams_dir"),
            normalize_rewards=env_config.get("normalize_rewards", True),
            server_config=env_config.get("server_config"),
            log_level=env_config.get("log_level", logging.DEBUG)
        )
        
        # 2. Create RL agent
        rl_agent = create_agent_from_config(
            env, rl_agent_config, model_state_dict, device_name
        )
        env.register_agent(rl_agent, env.agent_ids[0])
        
        # 3. Create opponent agent
        if episode_type == "self_play":
            # Self-play: create another RL agent with same model but no learning
            opponent_config = rl_agent_config.copy()
            opponent_config["learning"] = False
            opponent_agent = create_agent_from_config(
                env, opponent_config, model_state_dict, device_name
            )
            opponent_type = "self"
        else:
            # Regular opponent
            opponent_agent = create_agent_from_config(
                env, opponent_agent_config, None, device_name
            )
            opponent_type = opponent_agent_config.get("type", "unknown")
        
        env.register_agent(opponent_agent, env.agent_ids[1])
        
        # 4. Run episode (adapted from existing run_episode logic)
        if episode_type == "self_play":
            result = _run_self_play_episode(
                env, rl_agent, opponent_agent, gamma, lam, device_name, record_init
            )
        else:
            result = _run_opponent_episode(
                env, rl_agent, opponent_agent, gamma, lam, device_name, record_init, opponent_type
            )
        
        # 5. Clean up environment
        env.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in process worker: {e}", exc_info=True)
        raise


def _run_opponent_episode(
    env: PokemonEnv,
    rl_agent: RLAgent,
    opponent_agent: Any,
    gamma: float,
    lam: float,
    device_name: str,
    record_init: bool,
    opponent_type: str
) -> tuple:
    """Run episode with opponent (adapted from run_episode_with_opponent)."""
    observations, info, masks = env.reset(return_masks=True)
    obs0 = observations[env.agent_ids[0]]
    obs1 = observations[env.agent_ids[1]]
    mask0, mask1 = masks
    
    # Reset hidden states for LSTM networks at episode start
    rl_agent.reset_hidden_states()
    if hasattr(opponent_agent, 'reset_hidden_states'):
        opponent_agent.reset_hidden_states()

    init_tuple: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if info.get("request_teampreview"):
        order0 = rl_agent.choose_team(obs0)
        order1 = opponent_agent.choose_team(obs1)
        observations, *_, masks = env.step(
            {env.agent_ids[0]: order0, env.agent_ids[1]: order1}, return_masks=True
        )
        obs0 = observations[env.agent_ids[0]]
        obs1 = observations[env.agent_ids[1]]
        mask0, mask1 = masks

    episode_data = []
    obs = obs0
    mask = mask0
    total_reward = 0.0
    done = False

    device = get_device(prefer_gpu=False, device_name=device_name)

    while not done:
        # RL agent action
        if hasattr(rl_agent, 'select_action'):
            # For RL agents, use select_action which returns probabilities
            probs = rl_agent.select_action(obs, mask)
            rng = env.rng
            action = int(rng.choice(len(probs), p=probs))
            log_prob = float(np.log(probs[action] + 1e-8))
        else:
            # Fallback for other agent types
            action = rl_agent.act(obs, mask)
            log_prob = 0.0  # Default log prob for non-RL agents
        
        if record_init and init_tuple is None:
            init_tuple = (obs.copy(), mask.copy(), log_prob)

        # Opponent action
        if hasattr(opponent_agent, 'select_action'):
            opp_probs = opponent_agent.select_action(obs1, mask1)
            rng = env.rng
            opp_action = int(rng.choice(len(opp_probs), p=opp_probs))
        else:
            opp_action = opponent_agent.act(env.current_battle)

        # Environment step
        observations, rewards, terms, truncs, info, masks = env.step(
            {env.agent_ids[0]: action, env.agent_ids[1]: opp_action}, return_masks=True
        )
        done = any(terms.values()) or any(truncs.values())

        reward = rewards[env.agent_ids[0]]
        total_reward += reward

        # Store transition
        episode_data.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "log_prob": log_prob,
            "mask": mask,
        })

        # Update for next step
        obs = observations[env.agent_ids[0]]
        obs1 = observations[env.agent_ids[1]]
        mask0, mask1 = masks
        mask = mask0

    # Convert to batch format
    batch = _convert_episode_to_batch(episode_data, rl_agent, device, gamma, lam)
    sub_totals = env._sub_reward_logs.copy() if hasattr(env, '_sub_reward_logs') else {}
    
    return (batch, total_reward, init_tuple, sub_totals, opponent_type)


def _run_self_play_episode(
    env: PokemonEnv,
    rl_agent: RLAgent,
    opponent_agent: RLAgent,
    gamma: float,
    lam: float,
    device_name: str,
    record_init: bool
) -> tuple:
    """Run self-play episode (adapted from run_episode)."""
    observations, info, masks = env.reset(return_masks=True)
    obs0 = observations[env.agent_ids[0]]
    obs1 = observations[env.agent_ids[1]]
    mask0, mask1 = masks

    # Reset hidden states
    rl_agent.reset_hidden_states()
    opponent_agent.reset_hidden_states()

    init_tuple: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if info.get("request_teampreview"):
        order0 = rl_agent.choose_team(obs0)
        order1 = opponent_agent.choose_team(obs1)
        observations, *_, masks = env.step(
            {env.agent_ids[0]: order0, env.agent_ids[1]: order1}, return_masks=True
        )
        obs0 = observations[env.agent_ids[0]]
        obs1 = observations[env.agent_ids[1]]
        mask0, mask1 = masks

    episode_data = []
    obs = obs0
    mask = mask0
    total_reward = 0.0
    done = False

    device = get_device(prefer_gpu=False, device_name=device_name)

    while not done:
        # Both agents act
        probs0 = rl_agent.select_action(obs0, mask0)
        probs1 = opponent_agent.select_action(obs1, mask1)
        rng = env.rng
        action0 = int(rng.choice(len(probs0), p=probs0))
        action1 = int(rng.choice(len(probs1), p=probs1))
        log_prob0 = float(np.log(probs0[action0] + 1e-8))
        log_prob1 = float(np.log(probs1[action1] + 1e-8))
        
        if record_init and init_tuple is None:
            init_tuple = (obs0.copy(), mask0.copy(), log_prob0)

        # Environment step
        observations, rewards, terms, truncs, info, masks = env.step(
            {env.agent_ids[0]: action0, env.agent_ids[1]: action1}, return_masks=True
        )
        done = any(terms.values()) or any(truncs.values())

        reward = rewards[env.agent_ids[0]]
        total_reward += reward

        # Store transition for main agent
        episode_data.append({
            "obs": obs0,
            "action": action0,
            "reward": reward,
            "log_prob": log_prob0,
            "mask": mask0,
        })

        # Update for next step
        obs0 = observations[env.agent_ids[0]]
        obs1 = observations[env.agent_ids[1]]
        mask0, mask1 = masks

    # Convert to batch format
    batch = _convert_episode_to_batch(episode_data, rl_agent, device, gamma, lam)
    sub_totals = env._sub_reward_logs.copy() if hasattr(env, '_sub_reward_logs') else {}
    
    return (batch, total_reward, init_tuple, sub_totals, "self")


def _convert_episode_to_batch(
    episode_data: list,
    agent: RLAgent,
    device: torch.device,
    gamma: float,
    lam: float
) -> dict:
    """Convert episode data to batch format."""
    if not episode_data:
        raise ValueError("Episode data is empty")

    # Extract data
    observations = np.stack([step["obs"] for step in episode_data])
    actions = np.array([step["action"] for step in episode_data])
    rewards = np.array([step["reward"] for step in episode_data])
    log_probs = np.stack([step["log_prob"] for step in episode_data])
    masks = np.stack([step["mask"] for step in episode_data])

    # Compute values
    obs_tensors = torch.from_numpy(observations).float().to(device)
    
    # Get values from agent
    with torch.no_grad():
        if hasattr(agent, 'get_value'):
            values = []
            for obs_tensor in obs_tensors:
                value = agent.get_value(obs_tensor)
                values.append(value)
            values = torch.stack(values).cpu().numpy()
        else:
            # Fallback for basic networks
            values = agent.value_net(obs_tensors).cpu().numpy().flatten()

    # Compute advantages using GAE
    advantages, returns = compute_gae(rewards, values, gamma, lam)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "old_log_probs": log_probs,
        "advantages": advantages,
        "returns": returns,
        "action_masks": masks,
        "episode_lengths": np.array([len(episode_data)], dtype=np.int64),
    }