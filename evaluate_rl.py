from __future__ import annotations

import argparse
import logging
import sys
import yaml
from pathlib import Path
from datetime import datetime

# Repository root path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded configuration from: %s", config_path)
        return data
    except Exception as e:
        logger.warning("Could not load config from %s: %s", config_path, e)
        return {}


def setup_logging(log_dir: str, params: dict[str, object]) -> None:
    """Set up file logging and write parameters."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_path / f"eval_{timestamp}.log", encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logging.info("Run parameters: %s", params)


# Use poke_env from .venv instead of copy_of_poke-env

from src.env.wrappers import SingleAgentCompatibilityWrapper  # noqa: E402
from src.env.pokemon_env import PokemonEnv  # noqa: E402
from src.state.state_observer import StateObserver  # noqa: E402
from src.action import action_helper  # noqa: E402
from src.agents import PolicyNetwork, ValueNetwork, RLAgent  # noqa: E402
from src.agents.random_agent import RandomAgent  # noqa: E402
from src.agents.rule_based_player import RuleBasedPlayer  # noqa: E402
from src.bots import RandomBot, MaxDamageBot  # noqa: E402
from src.agents.network_factory import create_policy_network, create_value_network, get_network_info  # noqa: E402
from src.utils.device_utils import get_device, transfer_to_device, get_device_info  # noqa: E402
from src.algorithms import PPOAlgorithm  # noqa: E402
from src.teams.team_loader import TeamLoader  # noqa: E402
from src.utils.action_probability_logger import ActionProbabilityLogger  # noqa: E402
import torch  # noqa: E402
from torch import optim  # noqa: E402


def init_env(*, save_replays: bool | str = False, single: bool = True, player_names: tuple[str, str] | None = None, team_mode: str = "default", teams_dir: str | None = None):
    """Create ``PokemonEnv`` for evaluation."""
    import time
    import random

    # プレイヤー名に一意性を確保するためのタイムスタンプとランダム数を追加
    if player_names:
        timestamp = str(int(time.time() * 1000))[-6:]  # 最後の6桁のタイムスタンプ
        random_num = random.randint(10, 99)  # 2桁のランダム数
        unique_suffix = f"{timestamp}{random_num}"
        
        unique_names = (
            f"{player_names[0][:10]}_{unique_suffix}"[:18],  # 18文字制限
            f"{player_names[1][:10]}_{unique_suffix}"[:18]   # 18文字制限
        )
    else:
        unique_names = None

    # Setup team loader if using random teams
    team_loader = None
    if team_mode == "random":
        if teams_dir is None:
            teams_dir = str(ROOT_DIR / "config" / "teams")
        team_loader = TeamLoader(teams_dir)

    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
        save_replays=save_replays,
        player_names=unique_names,
        team_mode=team_mode,
        teams_dir=teams_dir,
        team_loader=team_loader,
    )
    if single:
        return SingleAgentCompatibilityWrapper(env)
    return env




def run_episode(agent: RLAgent) -> tuple[bool, float]:
    """Run one battle and return win flag and total reward."""

    env = agent.env
    
    # Reset hidden states at episode start
    if hasattr(agent, 'reset_hidden_states'):
        agent.reset_hidden_states()
    
    obs, info, action_mask = env.reset(return_masks=True)
    if info.get("request_teampreview"):
        team_cmd = agent.choose_team(obs)
        obs, action_mask, _, done, _ = env.step(team_cmd, return_masks=True)
    else:
        done = False
    total_reward = 0.0
    while not done:
        action = agent.act(obs, action_mask)
        obs, action_mask, reward, done, _ = env.step(action, return_masks=True)
        total_reward += float(reward)
    won = env.env._env_players[env.env.agent_ids[0]].n_won_battles == 1
    return won, total_reward


def run_episode_multi(
    agent0: RLAgent, agent1: RLAgent, action_logger=None
) -> tuple[bool, bool, float, float]:
    """Run one battle between two agents and return win flags and rewards."""

    env = agent0.env
    
    # Reset hidden states at episode start
    if hasattr(agent0, 'reset_hidden_states'):
        agent0.reset_hidden_states()
    if hasattr(agent1, 'reset_hidden_states'):
        agent1.reset_hidden_states()
    
    observations, info, masks = env.reset(return_masks=True)
    obs0 = observations[env.agent_ids[0]]
    obs1 = observations[env.agent_ids[1]]
    mask0, mask1 = masks

    if info.get("request_teampreview"):
        order0 = agent0.choose_team(obs0)
        order1 = agent1.choose_team(obs1)
        observations, *_ , masks = env.step(
            {"player_0": order0, "player_1": order1}, return_masks=True
        )
        obs0 = observations[env.agent_ids[0]]
        obs1 = observations[env.agent_ids[1]]
        mask0, mask1 = masks

    done = False
    reward0 = 0.0
    reward1 = 0.0
    turn = 0
    while not done:
        turn += 1
        action0 = agent0.act(obs0, mask0) if env._need_action[env.agent_ids[0]] else 0
        action1 = agent1.act(obs1, mask1) if env._need_action[env.agent_ids[1]] else 0
        
        # Log action probabilities if logger is provided
        if action_logger and isinstance(agent0, RLAgent):
            probs_data = agent0.get_last_action_probs()
            if probs_data:
                probs, mask = probs_data
                # Get current battle object
                battle = env.get_current_battle(env.agent_ids[0])
                action_logger.log_turn(env.agent_ids[0], turn, probs, mask, action0, battle)
                
        observations, rewards, terms, truncs, _, masks = env.step(
            {"player_0": action0, "player_1": action1}, return_masks=True
        )
        mask0, mask1 = masks
        obs0 = observations[env.agent_ids[0]]
        obs1 = observations[env.agent_ids[1]]
        reward0 += float(rewards[env.agent_ids[0]])
        reward1 += float(rewards[env.agent_ids[1]])
        done = terms[env.agent_ids[0]] or truncs[env.agent_ids[0]]

    win0 = env._env_players[env.agent_ids[0]].n_won_battles == 1
    win1 = env._env_players[env.agent_ids[1]].n_won_battles == 1
    return win0, win1, reward0, reward1


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


def evaluate_single(
    model_path: str, n: int = 1, replay_dir: str | bool = "replays", opponent: str = "random", team: str = "default", teams_dir: str | None = None, device: str = "auto", log_action_probs: bool = False
) -> None:
    # Multi-agent環境でRL学習済みエージェント vs 指定した対戦相手
    model_name = Path(model_path).stem
    
    # 対戦相手の分かりやすい名前を生成
    opponent_names = {
        "random": "RandomBot",
        "max": "MaxDamageBot",
        "rule": "RuleBasedPlayer"
    }
    opponent_name = opponent_names.get(opponent, f"{opponent.capitalize()}Agent")
    
    # Team configuration
    team_mode = team
    if team == "random":
        if teams_dir is None:
            teams_dir = str(ROOT_DIR / "config" / "teams")
        logger.info("Using random team mode with teams from: %s", teams_dir)
    else:
        team_mode = "default"
        logger.info("Using default team mode")
    
    # Setup device
    device_obj = get_device(prefer_gpu=True, device_name=device)
    device_info = get_device_info(device_obj)
    logger.info("Device info: %s", device_info)
    
    # モデルの準備
    state_dict = torch.load(model_path, map_location="cpu")

    # Initialize action probability logger if requested
    action_logger = None
    if log_action_probs:
        action_logger = ActionProbabilityLogger("logs/evaluate", model_name)
        logger.info("Action probability logging enabled")
    
    wins = 0
    total_reward = 0.0
    for i in range(n):
        # 各バトルごとに新しい環境を作成（ユニークな名前で）
        env = init_env(save_replays=replay_dir, single=False, player_names=(model_name, opponent_name), team_mode=team_mode, teams_dir=teams_dir)
        
        # Check if state_dict contains network configuration
        network_config = {}
        if isinstance(state_dict, dict) and "network_config" in state_dict:
            network_config = state_dict["network_config"]
            logger.info("Found network config: %s", network_config)
        else:
            # Try to detect network type from state_dict structure
            if isinstance(state_dict, dict) and "policy" in state_dict:
                policy_keys = list(state_dict["policy"].keys())
                
                # Check input dimension to determine if move embeddings are used
                input_dim = None
                if "model.0.weight" in state_dict["policy"]:
                    input_dim = state_dict["policy"]["model.0.weight"].shape[1]
                    logger.info("Detected input dimension: %d", input_dim)
                
                # Determine hidden size from actual layer dimensions
                hidden_size = 128  # default
                
                # For Attention networks, get hidden size from input_proj layer
                if any("input_proj" in key for key in policy_keys):
                    if "input_proj.weight" in state_dict["policy"]:
                        hidden_size = state_dict["policy"]["input_proj.weight"].shape[0]
                        logger.info("Detected hidden_size from input_proj: %d", hidden_size)
                # For basic networks, get hidden size from first model layer
                elif "model.0.weight" in state_dict["policy"]:
                    hidden_size = state_dict["policy"]["model.0.weight"].shape[0]
                    logger.info("Detected hidden_size from model.0: %d", hidden_size)
                
                if any("input_proj" in key for key in policy_keys):
                    # This is an AttentionNetwork (with or without LSTM)
                    has_attention_layers = any("attention" in key for key in policy_keys)
                    network_config = {
                        "type": "attention",
                        "hidden_size": hidden_size,
                        "use_attention": has_attention_layers,
                        "use_lstm": any("lstm" in key for key in policy_keys),
                        "use_2layer": True
                    }
                    logger.info("Detected Attention network (with LSTM) from state_dict structure")
                elif any("attention" in key for key in policy_keys):
                    network_config = {
                        "type": "attention",
                        "hidden_size": hidden_size,
                        "use_attention": True,
                        "use_lstm": False,
                        "use_2layer": True
                    }
                    logger.info("Detected Attention network from state_dict structure")
                else:
                    # Basic network - check if it's designed for move embeddings
                    network_config = {
                        "type": "basic",
                        "hidden_size": hidden_size,
                        "use_lstm": False,
                        "use_attention": False,
                        "use_2layer": True
                    }
                    logger.info("Detected basic network from state_dict structure")
            elif isinstance(state_dict, dict):
                # Direct state_dict (old format)
                direct_keys = list(state_dict.keys())
                
                # Check input dimension
                input_dim = None
                if "model.0.weight" in state_dict:
                    input_dim = state_dict["model.0.weight"].shape[1]
                    logger.info("Detected input dimension: %d", input_dim)
                
                # Determine hidden size
                hidden_size = 256 if "model.0.weight" in state_dict and state_dict["model.0.weight"].shape[0] == 256 else 128
                
                if any("input_proj" in key for key in direct_keys):
                    # This is an AttentionNetwork (with or without LSTM)
                    has_attention_layers = any("attention" in key for key in direct_keys)
                    network_config = {
                        "type": "attention",
                        "hidden_size": hidden_size,
                        "use_attention": has_attention_layers,
                        "use_lstm": any("lstm" in key for key in direct_keys),
                        "use_2layer": True
                    }
                    logger.info("Detected Attention network (with LSTM) from direct state_dict")
                elif any("lstm" in key for key in direct_keys):
                    network_config = {
                        "type": "lstm",
                        "hidden_size": hidden_size,
                        "lstm_hidden_size": hidden_size,
                        "use_lstm": True,
                        "use_2layer": True
                    }
                    logger.info("Detected LSTM network from direct state_dict")
                else:
                    network_config = {
                        "type": "basic",
                        "hidden_size": hidden_size,
                        "use_lstm": False,
                        "use_attention": False,
                        "use_2layer": True
                    }
                    logger.info("Detected basic network from direct state_dict")
        
        logger.info("Using network config: %s", network_config)
        
        # RL学習済みエージェントの作成 (player_0)
        policy_net = create_policy_network(
            env.observation_space[env.agent_ids[0]], 
            env.action_space[env.agent_ids[0]],
            network_config
        )
        value_net = create_value_network(
            env.observation_space[env.agent_ids[0]],
            network_config
        )
        
        # Transfer to device
        policy_net = transfer_to_device(policy_net, device_obj)
        value_net = transfer_to_device(value_net, device_obj)
        
        # Load model weights
        if isinstance(state_dict, dict) and "policy" in state_dict and "value" in state_dict:
            policy_net.load_state_dict(state_dict["policy"])
            value_net.load_state_dict(state_dict["value"])
        else:
            policy_net.load_state_dict(state_dict)
        
        # Log network information
        policy_info = get_network_info(policy_net)
        value_info = get_network_info(value_net)
        logger.info("Policy network: %s", policy_info)
        logger.info("Value network: %s", value_info)
        
        params = list(policy_net.parameters()) + list(value_net.parameters())
        optimizer = optim.Adam(params, lr=1e-3)
        algorithm = PPOAlgorithm()  # Default algorithm for evaluation
        rl_agent = RLAgent(env, policy_net, value_net, optimizer, algorithm=algorithm)

        # 対戦相手エージェントの作成 (player_1)
        opponent_agent = create_opponent_agent(opponent, env)
        env.register_agent(opponent_agent, env.agent_ids[1])
        
        # Start battle logging if enabled
        if action_logger:
            action_logger.start_battle(i + 1, (model_name, opponent_name))
        
        win0, win1, reward0, reward1 = run_episode_multi(rl_agent, opponent_agent, action_logger)
        
        # End battle logging if enabled
        if action_logger:
            winner = model_name if win0 else opponent_name
            action_logger.end_battle(winner, {"player_0": reward0, "player_1": reward1})
        
        wins += int(win0)
        total_reward += reward0
        logger.info(
            "Battle %d %s_reward=%.2f win=%s %s_reward=%.2f win=%s", 
            i + 1, model_name, reward0, win0, opponent_name, reward1, win1
        )
        
        # バトル終了後に環境をクリーンアップ
        env.close()
    logger.info("Evaluation finished after %d battles", n)

    win_rate = wins / n if n else 0.0
    avg_reward = total_reward / n if n else 0.0
    logger.info("RL Agent vs %s - win_rate: %.2f avg_reward: %.2f", opponent, win_rate, avg_reward)
    
    # Save action probability logs if enabled
    if action_logger:
        action_logger.save()


def compare_models(
    model_a: str, model_b: str, n: int = 1, replay_dir: str | bool = "replays", team: str = "default", teams_dir: str | None = None, device: str = "auto"
) -> None:
    """Evaluate two models against each other and report win rates."""

    model_a_name = Path(model_a).stem
    model_b_name = Path(model_b).stem
    
    # Team configuration
    team_mode = team
    if team == "random":
        if teams_dir is None:
            teams_dir = str(ROOT_DIR / "config" / "teams")
        logger.info("Using random team mode with teams from: %s", teams_dir)
    else:
        team_mode = "default"
        logger.info("Using default team mode")
    
    # Setup device
    device_obj = get_device(prefer_gpu=True, device_name=device)
    device_info = get_device_info(device_obj)
    logger.info("Device info: %s", device_info)
    
    # モデルの準備
    state_dict0 = torch.load(model_a, map_location="cpu")
    state_dict1 = torch.load(model_b, map_location="cpu")

    wins0 = 0
    wins1 = 0
    total0 = 0.0
    total1 = 0.0
    for i in range(n):
        # 各バトルごとに新しい環境を作成（ユニークな名前で）
        env = init_env(save_replays=replay_dir, single=False, player_names=(model_a_name, model_b_name), team_mode=team_mode, teams_dir=teams_dir)

        # Get network configurations with auto-detection
        def detect_network_config(state_dict, model_name):
            if isinstance(state_dict, dict) and "network_config" in state_dict:
                return state_dict["network_config"]
            elif isinstance(state_dict, dict) and "policy" in state_dict:
                policy_keys = list(state_dict["policy"].keys())
                
                # Check input dimension and hidden size
                input_dim = None
                hidden_size = 128
                if "model.0.weight" in state_dict["policy"]:
                    input_dim = state_dict["policy"]["model.0.weight"].shape[1]
                    hidden_size = state_dict["policy"]["model.0.weight"].shape[0]
                    logger.info("%s - Detected input dimension: %d, hidden size: %d", model_name, input_dim, hidden_size)
                
                if any("input_proj" in key for key in policy_keys):
                    # AttentionNetwork
                    has_attention_layers = any("attention" in key for key in policy_keys)
                    return {
                        "type": "attention",
                        "hidden_size": hidden_size,
                        "use_attention": has_attention_layers,
                        "use_lstm": any("lstm" in key for key in policy_keys),
                        "use_2layer": True
                    }
                elif any("lstm" in key for key in policy_keys):
                    return {
                        "type": "lstm",
                        "hidden_size": hidden_size,
                        "lstm_hidden_size": hidden_size,
                        "use_lstm": True,
                        "use_2layer": True
                    }
                elif any("attention" in key for key in policy_keys):
                    return {
                        "type": "attention",
                        "hidden_size": hidden_size,
                        "use_attention": True,
                        "use_lstm": False,
                        "use_2layer": True
                    }
            return {
                "type": "basic",
                "hidden_size": hidden_size if 'hidden_size' in locals() else 128,
                "use_lstm": False,
                "use_attention": False,
                "use_2layer": True
            }
        
        network_config0 = detect_network_config(state_dict0, model_a_name)
        network_config1 = detect_network_config(state_dict1, model_b_name)
        logger.info("Model A network config: %s", network_config0)
        logger.info("Model B network config: %s", network_config1)

        # Create player_1 agent first so that registration order is correct
        policy1 = create_policy_network(
            env.observation_space[env.agent_ids[1]], 
            env.action_space[env.agent_ids[1]],
            network_config1
        )
        value1 = create_value_network(
            env.observation_space[env.agent_ids[1]],
            network_config1
        )
        
        # Transfer to device
        policy1 = transfer_to_device(policy1, device_obj)
        value1 = transfer_to_device(value1, device_obj)
        
        if isinstance(state_dict1, dict) and "policy" in state_dict1 and "value" in state_dict1:
            policy1.load_state_dict(state_dict1["policy"])
            value1.load_state_dict(state_dict1["value"])
        else:
            policy1.load_state_dict(state_dict1)
        params1 = list(policy1.parameters()) + list(value1.parameters())
        opt1 = optim.Adam(params1, lr=1e-3)
        algorithm = PPOAlgorithm()  # Default algorithm for evaluation
        agent1 = RLAgent(env, policy1, value1, opt1, algorithm=algorithm)
        env.register_agent(agent1, env.agent_ids[1])

        policy0 = create_policy_network(
            env.observation_space[env.agent_ids[0]], 
            env.action_space[env.agent_ids[0]],
            network_config0
        )
        value0 = create_value_network(
            env.observation_space[env.agent_ids[0]],
            network_config0
        )
        
        # Transfer to device
        policy0 = transfer_to_device(policy0, device_obj)
        value0 = transfer_to_device(value0, device_obj)
        
        if isinstance(state_dict0, dict) and "policy" in state_dict0 and "value" in state_dict0:
            policy0.load_state_dict(state_dict0["policy"])
            value0.load_state_dict(state_dict0["value"])
        else:
            policy0.load_state_dict(state_dict0)
        params0 = list(policy0.parameters()) + list(value0.parameters())
        opt0 = optim.Adam(params0, lr=1e-3)
        algorithm = PPOAlgorithm()  # Default algorithm for evaluation
        agent0 = RLAgent(env, policy0, value0, opt0, algorithm=algorithm)

        win0, win1, reward0, reward1 = run_episode_multi(agent0, agent1)
        wins0 += int(win0)
        wins1 += int(win1)
        total0 += reward0
        total1 += reward1
        logger.info(
            "Battle %d %s_reward=%.2f win=%s %s_reward=%.2f win=%s",
            i + 1,
            model_a_name,
            reward0,
            win0,
            model_b_name,
            reward1,
            win1,
        )
        
        # バトル終了後に環境をクリーンアップ
        env.close()
    win_rate0 = wins0 / n if n else 0.0
    win_rate1 = wins1 / n if n else 0.0
    avg0 = total0 / n if n else 0.0
    avg1 = total1 / n if n else 0.0
    logger.info("%s win_rate: %.2f avg_reward: %.2f", model_a_name, win_rate0, avg0)
    logger.info("%s win_rate: %.2f avg_reward: %.2f", model_b_name, win_rate1, avg1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Evaluate trained RL model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yml",
        help="path to configuration file",
    )
    parser.add_argument("--model", type=str, help="path to model file (.pt)")
    parser.add_argument(
        "--models",
        nargs=2,
        metavar=("A", "B"),
        help="two model files for head-to-head evaluation",
    )
    parser.add_argument("--n", type=int, default=None, help="number of battles")
    parser.add_argument(
        "--replay-dir",
        type=str,
        default=None,
        help="directory to save battle replays",
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "max", "rule"],
        default=None,
        help="opponent type for single model evaluation (random, max, or rule)",
    )
    parser.add_argument(
        "--team",
        type=str,
        choices=["default", "random"],
        default=None,
        help="team selection mode (default or random)",
    )
    parser.add_argument(
        "--teams-dir",
        type=str,
        default=None,
        help="directory containing team files for random team mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to use for evaluation (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--log-action-probs",
        action="store_true",
        help="log action probabilities for each turn to logs/evaluate directory",
    )
    args = parser.parse_args()

    # Load configuration file
    config = load_config(args.config)

    # Apply defaults from config file, then override with command line arguments
    n = args.n if args.n is not None else config.get("episodes", 1)
    replay_dir = args.replay_dir if args.replay_dir is not None else config.get("replay_dir", "replays")
    opponent = args.opponent if args.opponent is not None else config.get("opponent", "random")
    team = args.team if args.team is not None else config.get("team", "default")
    teams_dir = args.teams_dir if args.teams_dir is not None else config.get("teams_dir", None)
    device = args.device if args.device is not None else config.get("device", "auto")

    # Setup logging with final parameters
    final_params = {
        "config": args.config,
        "model": args.model,
        "models": args.models,
        "n": n,
        "replay_dir": replay_dir,
        "opponent": opponent,
        "team": team,
        "teams_dir": teams_dir,
        "device": device,
        "log_action_probs": args.log_action_probs,
    }
    setup_logging("logs", final_params)

    if args.models:
        compare_models(args.models[0], args.models[1], n, replay_dir, team, teams_dir, device)
    elif args.model:
        evaluate_single(args.model, n, replay_dir, opponent, team, teams_dir, device, args.log_action_probs)
    else:
        parser.error("--model or --models is required")
