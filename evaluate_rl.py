from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Repository root path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)


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


# Ensure bundled poke_env package is importable
POKE_ENV_DIR = ROOT_DIR / "copy_of_poke-env"
if str(POKE_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(POKE_ENV_DIR))

from src.env.wrappers import SingleAgentCompatibilityWrapper  # noqa: E402
from src.env.pokemon_env import PokemonEnv  # noqa: E402
from src.state.state_observer import StateObserver  # noqa: E402
from src.action import action_helper  # noqa: E402
from src.agents import PolicyNetwork, ValueNetwork, RLAgent  # noqa: E402
from src.agents.random_agent import RandomAgent  # noqa: E402
from src.agents.rule_based_player import RuleBasedPlayer  # noqa: E402
from src.bots import RandomBot, MaxDamageBot  # noqa: E402
import torch  # noqa: E402
from torch import optim  # noqa: E402


def init_env(*, save_replays: bool | str = False, single: bool = True):
    """Create ``PokemonEnv`` for evaluation."""

    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
        save_replays=save_replays,
    )
    if single:
        return SingleAgentCompatibilityWrapper(env)
    return env


def set_player_names(env, player_names: tuple[str, str]):
    """Set custom player names for environment players."""
    from poke_env.ps_client.account_configuration import AccountConfiguration
    
    # 元のreset関数を保存
    if not hasattr(env, '_original_reset'):
        env._original_reset = env.reset
    
    def custom_reset(*args, **kwargs):
        # 元のresetを実行
        result = env._original_reset(*args, **kwargs)
        
        # reset後にプレイヤー名を設定
        if hasattr(env, '_env_players'):
            if 'player_0' in env._env_players:
                env._env_players['player_0'].account_configuration = AccountConfiguration(player_names[0], None)
            if 'player_1' in env._env_players:
                env._env_players['player_1'].account_configuration = AccountConfiguration(player_names[1], None)
        
        return result
    
    # reset関数を置き換え
    env.reset = custom_reset


def run_episode(agent: RLAgent) -> tuple[bool, float]:
    """Run one battle and return win flag and total reward."""

    env = agent.env
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
    agent0: RLAgent, agent1: RLAgent, player_names: tuple[str, str] = None
) -> tuple[bool, bool, float, float]:
    """Run one battle between two agents and return win flags and rewards."""

    env = agent0.env
    
    # プレイヤー名を設定（reset前に行う）
    if player_names:
        set_player_names(env, player_names)
    
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
    while not done:
        action0 = agent0.act(obs0, mask0) if env._need_action[env.agent_ids[0]] else 0
        action1 = agent1.act(obs1, mask1) if env._need_action[env.agent_ids[1]] else 0
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
    model_path: str, n: int = 1, replay_dir: str | bool = "replays", opponent: str = "random"
) -> None:
    # Multi-agent環境でRL学習済みエージェント vs 指定した対戦相手
    model_name = Path(model_path).stem
    opponent_name = f"{opponent.capitalize()}Agent"
    env = init_env(save_replays=replay_dir, single=False)
    
    # RL学習済みエージェントの作成 (player_0)
    policy_net = PolicyNetwork(
        env.observation_space[env.agent_ids[0]], env.action_space[env.agent_ids[0]]
    )
    value_net = ValueNetwork(env.observation_space[env.agent_ids[0]])
    state_dict = torch.load(model_path, map_location="cpu")
    if isinstance(state_dict, dict) and "policy" in state_dict and "value" in state_dict:
        policy_net.load_state_dict(state_dict["policy"])
        value_net.load_state_dict(state_dict["value"])
    else:
        policy_net.load_state_dict(state_dict)
    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    rl_agent = RLAgent(env, policy_net, value_net, optimizer)

    # 対戦相手エージェントの作成 (player_1)
    opponent_agent = create_opponent_agent(opponent, env)
    env.register_agent(opponent_agent, env.agent_ids[1])

    wins = 0
    total_reward = 0.0
    for i in range(n):
        win0, win1, reward0, reward1 = run_episode_multi(rl_agent, opponent_agent, (model_name, opponent_name))
        wins += int(win0)
        total_reward += reward0
        logger.info(
            "Battle %d %s_reward=%.2f win=%s %s_reward=%.2f win=%s", 
            i + 1, model_name, reward0, win0, opponent_name, reward1, win1
        )

    env.close()
    logger.info("Evaluation finished after %d battles", n)

    win_rate = wins / n if n else 0.0
    avg_reward = total_reward / n if n else 0.0
    logger.info("RL Agent vs %s - win_rate: %.2f avg_reward: %.2f", opponent, win_rate, avg_reward)


def compare_models(
    model_a: str, model_b: str, n: int = 1, replay_dir: str | bool = "replays"
) -> None:
    """Evaluate two models against each other and report win rates."""

    model_a_name = Path(model_a).stem
    model_b_name = Path(model_b).stem
    env = init_env(save_replays=replay_dir, single=False)

    # Create player_1 agent first so that registration order is correct
    policy1 = PolicyNetwork(
        env.observation_space[env.agent_ids[1]], env.action_space[env.agent_ids[1]]
    )
    value1 = ValueNetwork(env.observation_space[env.agent_ids[1]])
    state_dict1 = torch.load(model_b, map_location="cpu")
    if isinstance(state_dict1, dict) and "policy" in state_dict1 and "value" in state_dict1:
        policy1.load_state_dict(state_dict1["policy"])
        value1.load_state_dict(state_dict1["value"])
    else:
        policy1.load_state_dict(state_dict1)
    params1 = list(policy1.parameters()) + list(value1.parameters())
    opt1 = optim.Adam(params1, lr=1e-3)
    agent1 = RLAgent(env, policy1, value1, opt1)
    env.register_agent(agent1, env.agent_ids[1])

    policy0 = PolicyNetwork(
        env.observation_space[env.agent_ids[0]], env.action_space[env.agent_ids[0]]
    )
    value0 = ValueNetwork(env.observation_space[env.agent_ids[0]])
    state_dict0 = torch.load(model_a, map_location="cpu")
    if isinstance(state_dict0, dict) and "policy" in state_dict0 and "value" in state_dict0:
        policy0.load_state_dict(state_dict0["policy"])
        value0.load_state_dict(state_dict0["value"])
    else:
        policy0.load_state_dict(state_dict0)
    params0 = list(policy0.parameters()) + list(value0.parameters())
    opt0 = optim.Adam(params0, lr=1e-3)
    agent0 = RLAgent(env, policy0, value0, opt0)

    wins0 = 0
    wins1 = 0
    total0 = 0.0
    total1 = 0.0
    for i in range(n):
        win0, win1, reward0, reward1 = run_episode_multi(agent0, agent1, (model_a_name, model_b_name))
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
    parser.add_argument("--model", type=str, help="path to model file (.pt)")
    parser.add_argument(
        "--models",
        nargs=2,
        metavar=("A", "B"),
        help="two model files for head-to-head evaluation",
    )
    parser.add_argument("--n", type=int, default=1, help="number of battles")
    parser.add_argument(
        "--replay-dir",
        type=str,
        default="replays",
        help="directory to save battle replays",
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "max", "rule"],
        default="random",
        help="opponent type for single model evaluation (random, max, or rule)",
    )
    args = parser.parse_args()

    setup_logging("logs", vars(args))

    if args.models:
        compare_models(args.models[0], args.models[1], args.n, args.replay_dir)
    elif args.model:
        evaluate_single(args.model, args.n, args.replay_dir, args.opponent)
    else:
        parser.error("--model or --models is required")
