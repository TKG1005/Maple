"""Gymnasium environment skeleton for Pokémon battles."""

from __future__ import annotations

from typing import Any, Tuple
import concurrent.futures as cf
import os
import warnings


import numpy as np

import gymnasium as gym
from gymnasium.spaces import Dict
import asyncio
import time
import sys
import random
import logging
from pathlib import Path
import yaml
from poke_env.concurrency import POKE_LOOP

from .env_player import EnvPlayer
from .dual_mode_player import DualModeEnvPlayer, validate_mode_configuration
from src.rewards import HPDeltaReward, CompositeReward, RewardNormalizer
from src.sim.battle_state_serializer import BattleStateManager, PokeEnvBattleSerializer
from src.env.rqid_notifier import get_global_rqid_notifier
from src.profiling.metrics import emit_metric


class PokemonEnv(gym.Env):
    """A placeholder Gymnasium environment for Pokémon battles."""

    metadata = {"render_modes": [None]}

    def __init__(
        self,
        state_observer: Any,
        action_helper: Any,
        opponent_player: Any | None = None,
        *,
        seed: int | None = None,
        save_replays: bool | str = False,
        reward: str = "composite",
        reward_config_path: str | None = None,
        player_names: tuple[str, str] | None = None,
        team_mode: str = "default",
        teams_dir: str | None = None,
        team_loader: Any = None,
        normalize_rewards: bool = True,
        server_configuration: Any = None,
        battle_mode: str = "local",  # "local" or "online"
        log_level: int = logging.DEBUG,
        reuse_processes: bool | None = None,
        max_processes: int | None = None,
        **kwargs: Any,
        ) -> None:
        super().__init__()

        # "gen9bss"ルールでは行動空間は10で固定だったが、
        # Struggle 専用インデックスを追加して11に拡張
        self.ACTION_SIZE = 11
        self.MAX_TURNS = 1000  # エピソードの最大ターン数

        # Step10: 非同期アクションキューを導入
        # 数値アクションだけでなく、チーム選択コマンドなどの文字列も
        # 取り扱えるよう ``Any`` 型のキューを使用する
        self._action_queues: dict[str, asyncio.Queue[Any]] = {
            agent_id: asyncio.Queue() for agent_id in ("player_0", "player_1")
        }
        # EnvPlayer から受け取る battle オブジェクト用キュー
        self._battle_queues: dict[str, asyncio.Queue[Any]] = {
            agent_id: asyncio.Queue() for agent_id in ("player_0", "player_1")
        }

        self._agents: dict[str, Any] = {}

        self.opponent_player = opponent_player
        self.state_observer = state_observer
        self.action_helper = action_helper
        self.rng = np.random.default_rng(seed)
        self.save_replays = save_replays
        self._logger = logging.getLogger(__name__)
        self.log_level = log_level
        try:
            # Ensure env logger honors requested level so diagnosis logs appear
            self._logger.setLevel(self.log_level)
        except Exception:
            pass
        self.reward_type = reward
        self.reward_config_path = reward_config_path
        self.player_names = player_names
        self.team_mode = team_mode
        self.teams_dir = teams_dir
        self.normalize_rewards = normalize_rewards
        self.server_configuration = server_configuration
        self.battle_mode = battle_mode
        self.reuse_processes = reuse_processes
        self.max_processes = max_processes
        # micro-sleep 撤廃に伴いトグルは廃止（Step4-2）
        
        # Skip validation for now as it requires complete configuration
        # TODO: Implement proper configuration validation in Phase 3
        # self._validate_battle_mode_config(config)
        
        # マルチエージェント用のエージェントID
        self.agent_ids = ("player_0", "player_1")
        
        # Initialize team loader for random team mode
        self._team_loader = team_loader
        if self._team_loader is None and self.team_mode == "random" and self.teams_dir:
            from src.teams import TeamLoader
            self._team_loader = TeamLoader(self.teams_dir)
            if self._team_loader.get_team_count() == 0:
                self._logger.warning("No teams loaded, falling back to default team")
                self.team_mode = "default"

        self._composite_rewards: dict[str, CompositeReward] = {}
        self._sub_reward_logs: dict[str, dict[str, float]] = {}
        
        # Initialize reward normalizers for each agent
        self._reward_normalizers: dict[str, RewardNormalizer] = {}
        if self.normalize_rewards:
            for agent_id in self.agent_ids:
                self._reward_normalizers[agent_id] = RewardNormalizer()

        # timeout 設定を config/env_config.yml から読み込む
        config_path = Path(__file__).resolve().parents[2] / "config" / "env_config.yml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.timeout = float(cfg.get("queue_timeout", 5))
        except Exception:
            self.timeout = 5.0

        # Determine the dimension of the observation vector from the
        # provided StateObserver and create the observation space.  The
        # StateObserver is expected to expose a `get_observation_dimension`
        # method which returns the length of the state vector.
        state_dim = self.state_observer.get_observation_dimension()
        single_obs_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(state_dim,),
            dtype=np.float32,
        )
        single_action_space = gym.spaces.Discrete(self.ACTION_SIZE)

        # 各エージェント用の観測・行動・報酬空間を Dict で保持
        self.observation_space = Dict(
            {agent_id: single_obs_space for agent_id in self.agent_ids}
        )
        self.action_space = Dict(
            {agent_id: single_action_space for agent_id in self.agent_ids}
        )
        self.reward_space = Dict(
            {
                agent_id: gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32)
                for agent_id in self.agent_ids
            }
        )

        # 最後に生成した行動マッピングを保持しておく
        self._action_mappings: dict[str, dict[int, tuple[str, str | int, bool]]] = {
            agent_id: {} for agent_id in self.agent_ids
        }

        # Player ごとの行動要求フラグ
        self._need_action: dict[str, bool] = {
            agent_id: False for agent_id in self.agent_ids
        }

        # チームプレビューで得た手持ちポケモン一覧
        self._team_rosters: dict[str, list[str]] = {
            agent_id: [] for agent_id in self.agent_ids
        }

        # Battle.last_request の更新を追跡するためのキャッシュ
        self._last_requests: dict[str, Any] = {
            agent_id: None for agent_id in self.agent_ids
        }
        # Track last seen rqid per player to detect request updates
        self._last_rqids: dict[str, int | None] = {
            agent_id: None for agent_id in self.agent_ids
        }
        # RQID used to build the last mask returned to the agent
        self._mask_rqids: dict[str, int | None] = {
            agent_id: None for agent_id in self.agent_ids
        }

        # HPDeltaReward をプレイヤーごとに保持
        self._hp_delta_rewards: dict[str, HPDeltaReward] = {}
        
        # Initialize battle state manager for Phase 3 serialization
        self._battle_serializer = PokeEnvBattleSerializer()
        self._state_manager = BattleStateManager(
            serializer=self._battle_serializer,
            storage_dir="battle_states"
        )
        # Per-player finished events to signal end of battle (WS and IPC unified)
        self._finished_events: dict[str, asyncio.Event] = {
            agent_id: asyncio.Event() for agent_id in self.agent_ids
        }
        
        # Feature toggles (can be reverted easily if needed)
        # Use per-step battle snapshot for mask computation
        self.use_snapshot_masks: bool = True
        try:
            env_flag = os.environ.get("MAPLE_USE_SNAPSHOT_MASKS")
            if env_flag is not None:
                self.use_snapshot_masks = env_flag.lower() in ("1", "true", "yes")
        except Exception:
            pass
        # Defer int->BattleOrder conversion to choose_move time
        self.defer_action_conversion: bool = True
        try:
            env_flag = os.environ.get("MAPLE_DEFER_CONVERSION")
            if env_flag is not None:
                self.defer_action_conversion = env_flag.lower() in ("1", "true", "yes")
        except Exception:
            pass

        # --- Diagnostics helpers (FD/handler stats) ---
        def _fd_count() -> int:
            try:
                return len(os.listdir('/dev/fd'))
            except Exception:
                return -1
        self._fd_count = _fd_count  # type: ignore[attr-defined]

    def _log_fd_stats(self, tag: str) -> None:
        """Log file descriptor and logger handler stats, plus IPC controller counts if available."""
        # Reduce verbosity: only key tags by default unless MAPLE_FD_VERBOSE=1
        try:
            verbose = str(os.environ.get("MAPLE_FD_VERBOSE", "0")).lower() in ("1", "true", "yes")
            essential = tag in {"reset_start", "env_close", "battle_finished_notify"}
            if not verbose and not essential:
                return
            fd = self._fd_count() if hasattr(self, "_fd_count") else -1
        except Exception:
            fd = -1
        try:
            root_handlers = len(logging.getLogger().handlers)
        except Exception:
            root_handlers = -1
        ipc_ctrls: dict[str, int] = {}
        try:
            for pid, p in getattr(self, "_env_players", {}).items():
                wrapper = getattr(p, "ipc_client_wrapper", None)
                if wrapper is not None:
                    ctrls = getattr(wrapper, "_controllers", {})
                    try:
                        ipc_ctrls[pid] = len(ctrls)  # type: ignore[arg-type]
                    except Exception:
                        ipc_ctrls[pid] = -1
        except Exception:
            pass
        try:
            self._logger.info(
                "[FD] tag=%s open_fds=%s root_handlers=%s ipc_controllers=%s",
                tag,
                fd,
                root_handlers,
                ipc_ctrls or {},
            )
        except Exception:
            pass
        
    # --- Robust WS teardown helper (online mode) ---
    def _force_close_psclient(self, ps_client: Any, player_id: str | None = None) -> None:
        try:
            # Stop listening if available
            stop_listening = getattr(ps_client, "stop_listening", None)
            if callable(stop_listening):
                try:
                    asyncio.run_coroutine_threadsafe(stop_listening(), POKE_LOOP).result()
                except Exception as e:
                    self._logger.exception(
                        "[WS_TEARDOWN] stop_listening failed pid=%s: %s",
                        player_id,
                        e,
                    )
                    raise
            # Close websocket and wait for closure
            ws = getattr(ps_client, "websocket", None)
            if ws is not None:
                try:
                    close_coro = getattr(ws, "close", None)
                    if callable(close_coro):
                        asyncio.run_coroutine_threadsafe(close_coro(), POKE_LOOP).result()
                except Exception as e:
                    self._logger.exception(
                        "[WS_TEARDOWN] websocket.close failed pid=%s: %s",
                        player_id,
                        e,
                    )
                    raise
                try:
                    wait_closed = getattr(ws, "wait_closed", None)
                    if callable(wait_closed):
                        asyncio.run_coroutine_threadsafe(wait_closed(), POKE_LOOP).result()
                except Exception as e:
                    self._logger.exception(
                        "[WS_TEARDOWN] websocket.wait_closed failed pid=%s: %s",
                        player_id,
                        e,
                    )
                    raise
                try:
                    setattr(ps_client, "websocket", None)
                except Exception as e:
                    self._logger.exception(
                        "[WS_TEARDOWN] clearing websocket ref failed pid=%s: %s",
                        player_id,
                        e,
                    )
                    raise
            # Cancel common background tasks if present
            for attr in ("_receive_task", "_keep_alive_task", "_listen_task"):
                try:
                    t = getattr(ps_client, attr, None)
                    if isinstance(t, asyncio.Task):
                        try:
                            t.cancel()
                        except Exception as e:
                            self._logger.exception(
                                "[WS_TEARDOWN] task cancel failed pid=%s (%s): %s",
                                player_id,
                                attr,
                                e,
                            )
                            raise
                except Exception as e:
                    self._logger.exception(
                        "[WS_TEARDOWN] access task attr failed pid=%s (%s): %s",
                        player_id,
                        attr,
                        e,
                    )
                    raise
        except Exception as e:
            self._logger.exception(
                "[WS_TEARDOWN] unexpected error pid=%s: %s", player_id, e
            )
            raise
        
    def _get_team_for_battle(self) -> str | None:
        """Get team content for the current battle.
        
        Returns
        -------
        str | None
            Team content in Pokemon Showdown format, or None if no team available
        """
        if self.team_mode == "random" and self._team_loader:
            # Get random team from loader
            team = self._team_loader.get_random_team()
            if team:
                # Extract first Pokemon name for logging
                first_line = team.split('\n')[0].strip()
                pokemon_name = first_line.split('@')[0].strip() if '@' in first_line else first_line
                self._logger.info("Selected random team starting with: %s", pokemon_name)
                return team
            else:
                self._logger.warning("No teams available from loader, falling back to default")
        
        # Default team loading
        team_path = Path(__file__).resolve().parents[2] / "config" / "my_team.txt"
        try:
            team = team_path.read_text(encoding="utf-8")
            self._logger.debug("Using default team from %s", team_path)
            return team
        except OSError:  # pragma: no cover - デバッグ用
            self._logger.warning("Default team file not found: %s", team_path)
            return None

    # ------------------------------------------------------------------
    # Agent interaction utilities
    # ------------------------------------------------------------------
    def register_agent(self, agent: Any, player_id: str = "player_0") -> None:
        """Register the controlling :class:`MapleAgent` for a given player."""
        if not hasattr(self, "_agents"):
            self._agents: dict[str, Any] = {}
        self._agents[player_id] = agent
        # Also store the identifier on the agent itself so that it can
        # reliably know which player it controls even if the internal mapping
        # is later overwritten.
        try:
            setattr(agent, "_player_id", player_id)
        except Exception:  # pragma: no cover - ignore if attribute cannot be set
            pass

    def get_current_battle(self, agent_id: str = "player_0") -> Any | None:
        """Return the latest :class:`Battle` object for ``agent_id``."""
        return getattr(self, "_current_battles", {}).get(agent_id)

    def get_action_mask(
        self, player_id: str, with_details: bool = False
    ) -> tuple[np.ndarray, dict[int, Any]]:
        """Return an action mask for ``player_id``.

        ``get_action_mask`` は非推奨です。基本的には ``step``/``reset`` の
        ``return_masks`` 機能を利用してください。

        When ``with_details`` is ``True`` the mapping contains human readable
        details provided by ``action_helper.get_available_actions_with_details``.
        The mask shows available actions based on the current battle state.
        """

        warnings.warn(
            "get_action_mask() is deprecated; use step(..., return_masks=True) instead",
            DeprecationWarning,
            stacklevel=2,
        )

        battle = self.get_current_battle(player_id)
        if battle is None:
            raise ValueError(f"No current battle for {player_id}")

        if with_details:
            mask, mapping = self.action_helper.get_available_actions_with_details(
                battle
            )
            self._action_mappings[player_id] = mapping
            return mask, mapping

        # without details use internal computation routine
        if self.use_snapshot_masks:
            battles_snapshot = {pid: self.get_current_battle(pid) for pid in self.agent_ids}
            masks = self._compute_all_masks(battles_snapshot)
        else:
            masks = self._compute_all_masks()
        idx = self.agent_ids.index(player_id)
        return masks[idx], self._action_mappings[player_id]

    def process_battle(self, battle: Any) -> int:
        """Create an observation and available action mask for ``battle``.

        The resulting state vector and action mask are sent to the registered
        :class:`MapleAgent` which returns an action index.
        """
        if not hasattr(self, "_agents") or not self._agents:
            raise RuntimeError("Agent not registered")

        observation = self.state_observer.observe(battle)

        # battle 情報から利用可能な行動マスクを生成
        action_mask, _ = self.action_helper.get_available_actions_with_details(battle)

        # ここでは player_0 のエージェントを利用する
        agent = self._agents.get("player_0")
        if agent is None:
            raise RuntimeError("Agent for player_0 not registered")
        action_idx = agent.select_action(observation, action_mask)
        return int(action_idx)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
        return_masks: bool = True,
    ) -> Tuple[Any, dict] | Tuple[Any, dict, Tuple[np.ndarray, np.ndarray]]:
        """Reset the environment and start a new battle.

        ``return_masks`` が ``True`` の場合、初期状態の行動マスクも返す。
        """

        super().reset(seed=seed)

        # 前回エピソードのキューをクリア
        self._action_queues = {agent_id: asyncio.Queue() for agent_id in self.agent_ids}
        self._battle_queues = {agent_id: asyncio.Queue() for agent_id in self.agent_ids}
        self._finished_events = {agent_id: asyncio.Event() for agent_id in self.agent_ids}
        self._need_action = {agent_id: True for agent_id in self.agent_ids}
        self._action_mappings = {agent_id: {} for agent_id in self.agent_ids}
        self._last_rqids = {agent_id: None for agent_id in self.agent_ids}
        self._logger.debug("environment reset: cleared action queues and mappings")
        # FD/handler snapshot at reset start
        self._log_fd_stats("reset_start")

        # poke_env は開発環境によってはインストールされていない場合があるため、
        # メソッド内で遅延インポートする。
        try:
            from poke_env.ps_client.server_configuration import (
                LocalhostServerConfiguration,
            )

        except Exception as exc:  # pragma: no cover - ランタイム用
            raise RuntimeError(
                "poke_env package is required to run PokemonEnv"
            ) from exc
        
        # Use the provided server configuration, or default to LocalhostServerConfiguration
        server_config = self.server_configuration if self.server_configuration is not None else LocalhostServerConfiguration
        

        # 対戦用のプレイヤーの処理
        # ランダムチームモードの場合は毎回新しいチームを選択するため、プレイヤーを再作成
        should_recreate_players = (
            self.team_mode == "random" and 
            hasattr(self, "_env_players") and 
            self._team_loader is not None
        )
        
        if not hasattr(self, "_env_players") or should_recreate_players:
            # 既存プレイヤーがある場合はクリーンアップ
            if hasattr(self, "_env_players"):
                for p in self._env_players.values():
                    if self.battle_mode == "online":
                        # Try player's own close_connection first
                        try:
                            if hasattr(p, "close_connection"):
                                asyncio.run_coroutine_threadsafe(
                                    p.close_connection(), POKE_LOOP
                                ).result()
                        except Exception as e:
                            self._logger.exception(
                                "[WS_CLEANUP] player.close_connection failed pid=%s: %s",
                                getattr(p, "player_id", "unknown"),
                                e,
                            )
                            raise
                        # Force-close only if websocket remains
                        try:
                            ps_client = getattr(p, "ps_client", None)
                            if ps_client is not None and getattr(ps_client, "websocket", None) is not None:
                                self._force_close_psclient(ps_client, getattr(p, "player_id", None))
                        except Exception as e:
                            self._logger.exception(
                                "[WS_CLEANUP] force_close_psclient failed pid=%s: %s",
                                getattr(p, "player_id", "unknown"),
                                e,
                            )
                            raise
                # Diagnostics: count remaining websockets after teardown (online only)
                try:
                    if self.battle_mode == "online":
                        rem = 0
                        for p in self._env_players.values():
                            pc = getattr(p, "ps_client", None)
                            if pc is not None and getattr(pc, "websocket", None) is not None:
                                rem += 1
                        if rem:
                            self._logger.info("[WS_CLEANUP] remaining_websockets=%d", rem)
                except Exception as e:
                    self._logger.exception(
                        "[WS_CLEANUP] remaining_websockets check failed: %s", e
                    )
            
            # 各プレイヤー用にチームを選択（ファイル指定 or random/default）
            override_team: str | None = None
            # 指定された team_mode が 'default'/'random' 以外の場合は、teams_dir からファイルを読み込む
            if self.team_mode not in ("default", "random") and self.teams_dir:
                try:
                    team_path = Path(self.teams_dir) / self.team_mode
                    content = team_path.read_text(encoding="utf-8").strip()
                    override_team = content if content else None
                    self._logger.info(f"Loaded team from file: {team_path.name}")
                except Exception as e:
                    self._logger.error(f"Failed to load team file '{self.team_mode}': {e}")
                    override_team = None
            # override_team があれば常にそれを使用、なければ既存のロジックで取得
            if override_team is not None:
                # 指定ファイルのチームを両方に適用
                team_player_0 = override_team
                team_player_1 = override_team
            else:
                # 各プレイヤーに独立してチームを割り当て（random または default）
                self._logger.info("Selecting team for player_0...")
                team_player_0 = self._get_team_for_battle()
                self._logger.info("Selecting team for player_1...")
                team_player_1 = self._get_team_for_battle()
            
            if team_player_0 is None:
                self._logger.warning("No team loaded for player_0, using None (may cause errors)")
                team_player_0 = None
            if team_player_1 is None:
                self._logger.warning("No team loaded for player_1, using None (may cause errors)")
                team_player_1 = None

            # プレイヤー名を設定（evaluate_rl.py用）
            from poke_env.ps_client.account_configuration import AccountConfiguration
            
            if self.player_names:
                # 18文字制限に合わせて名前を調整
                player_0_name = self.player_names[0][:18]
                account_config_0 = AccountConfiguration(player_0_name, None)
            else:
                account_config_0 = None

            # Create players based on battle mode
            self._env_players = {
                "player_0": self._create_battle_player(
                    "player_0",
                    server_config,
                    team_player_0,
                    account_config_0
                )
            }
            if self.opponent_player is None:
                if self.player_names:
                    # 18文字制限に合わせて名前を調整
                    player_1_name = self.player_names[1][:18]
                    account_config_1 = AccountConfiguration(player_1_name, None)
                else:
                    account_config_1 = None
                    
                self._env_players["player_1"] = self._create_battle_player(
                    "player_1",
                    server_config,
                    team_player_1,
                    account_config_1
                )
            else:
                self._env_players["player_1"] = self.opponent_player
        else:
            # 既存プレイヤーのバトル履歴をクリア（ランダムチーム以外）
            for p in self._env_players.values():
                if hasattr(p, "reset_battles"):
                    p.reset_battles()

        if self.opponent_player is not None and hasattr(
            self.opponent_player, "reset_battles"
        ):
            self.opponent_player.reset_battles()

        # Battle creation - unified for all modes (DualModeEnvPlayer handles WebSocket/IPC automatically)
        self._battle_task = asyncio.run_coroutine_threadsafe(
            self._run_battle(),
            POKE_LOOP,
        )

        # チーム選択リクエストを待機
        battle0 = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(self._battle_queues["player_0"].get(), self.timeout),
            POKE_LOOP,
        ).result()
        POKE_LOOP.call_soon_threadsafe(self._battle_queues["player_0"].task_done)
        battle1 = battle0
        if "player_1" in self._env_players:
            battle1 = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(self._battle_queues["player_1"].get(), self.timeout),
                POKE_LOOP,
            ).result()
            POKE_LOOP.call_soon_threadsafe(self._battle_queues["player_1"].task_done)

        # 各プレイヤーの手持ちポケモン種別を保存しておく
        self._team_rosters["player_0"] = [p.species for p in battle0.team.values()]
        self._team_rosters["player_1"] = [p.species for p in battle1.team.values()]

        self._current_battles = {"player_0": battle0, "player_1": battle1}

        # HPDeltaReward を初期化し、初期 HP を記録
        if self.reward_type == "hp_delta":
            self._hp_delta_rewards = {
                "player_0": HPDeltaReward(),
                "player_1": HPDeltaReward(),
            }
            self._hp_delta_rewards["player_0"].reset(battle0)
            self._hp_delta_rewards["player_1"].reset(battle1)
            self._composite_rewards = {}
            self._sub_reward_logs = {}
        elif self.reward_type == "composite":
            path = self.reward_config_path
            if path is None:
                path = str(
                    Path(__file__).resolve().parents[2]
                    / "config"
                    / "reward.yaml"
                )
            self._composite_rewards = {
                "player_0": CompositeReward(path),
                "player_1": CompositeReward(path),
            }
            self._composite_rewards["player_0"].reset(battle0)
            self._composite_rewards["player_1"].reset(battle1)
            self._hp_delta_rewards = {}
            self._sub_reward_logs = {pid: {} for pid in self.agent_ids}
        else:
            self._hp_delta_rewards = {}
            self._composite_rewards = {}
            self._sub_reward_logs = {}

        # Check if we're in teampreview phase
        if self._is_teampreview(battle0):
            self._logger.debug("In teampreview phase - using dummy observations")
            observation = {
                "player_0": self._get_teampreview_observation(),
                "player_1": self._get_teampreview_observation(),
            }
        else:
            observation = {
                "player_0": self.state_observer.observe(battle0),
                "player_1": self.state_observer.observe(battle1),
            }

        info: dict = {
            "battle_tag": battle0.battle_tag,
            "request_teampreview": True,
        }

        masks = self._compute_all_masks()
        self._last_requests = {
            pid: self._current_battles[pid].last_request for pid in self.agent_ids
        }

        if hasattr(self, "single_agent_mode"):
            if return_masks:
                return observation[self.agent_ids[0]], info, masks[0]
            return observation[self.agent_ids[0]], info

        if return_masks:
            return observation, info, masks
        return observation, info

    def _is_teampreview(self, battle: Any) -> bool:
        """Check teampreview strictly by absence of active_pokemon."""
        # 判定を単純化: アクティブが立っていない場合のみチームプレビューとみなす
        return battle.active_pokemon is None

    def _get_teampreview_observation(self) -> np.ndarray:
        """Generate dummy observation for teampreview phase."""
        # Always get current dimension from state_observer to ensure correctness
        obs_size = self.state_observer.get_observation_dimension()
        self._logger.debug(f"Generating teampreview dummy observation with {obs_size} dimensions")
        return np.zeros(obs_size, dtype=np.float32)

    async def _run_battle(self) -> None:
        """Start the battle coroutines concurrently."""

        await asyncio.gather(
            self._env_players["player_0"].battle_against(
                self._env_players["player_1"], n_battles=1
            ),
            self._env_players["player_1"].battle_against(
                self._env_players["player_0"], n_battles=1
            ),
        )

    def _notify_battle_finished(self, player_id: str, battle: Any) -> None:
        """Notify environment that player's battle has finished (WS/IPCsafe).

        This method is idempotent and schedules updates on POKE_LOOP where needed.
        """
        try:
            # Initialize once-guard map lazily
            if not hasattr(self, "_finished_enqueued"):
                # { player_id: set(battle_tag) } to prevent duplicate queue puts
                self._finished_enqueued: dict[str, set[str]] = {pid: set() for pid in self.agent_ids}
            # Update current battle reference
            self._current_battles[player_id] = battle
            # Also enqueue the latest battle snapshot exactly once per (pid, battle)
            try:
                tag = getattr(battle, "battle_tag", None)
                already = tag in self._finished_enqueued.get(player_id, set())
            except Exception:
                tag = None
                already = False
            if not already:
                # If currently on POKE_LOOP, perform immediate put/set for strict ordering.
                on_loop = False
                try:
                    loop = asyncio.get_running_loop()
                    on_loop = loop is POKE_LOOP
                except Exception:
                    on_loop = False
                if on_loop:
                    try:
                        self._battle_queues[player_id].put_nowait(battle)
                        try:
                            self._logger.debug(
                                "[ENDSIG-PUT] %s ts=%.6f on_loop=True mode=direct qsize=%d",
                                player_id,
                                time.monotonic(),
                                self._battle_queues[player_id].qsize(),
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass
                    try:
                        self._finished_enqueued[player_id].add(tag)
                    except Exception:
                        pass
                    try:
                        self._finished_events[player_id].set()
                        try:
                            self._logger.debug(
                                "[ENDSIG-SET] %s ts=%.6f on_loop=True",
                                player_id,
                                time.monotonic(),
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass
                else:
                    # Schedule on POKE_LOOP thread-safely (ordering: put -> set)
                    try:
                        ts_put = time.monotonic()
                        POKE_LOOP.call_soon_threadsafe(
                            self._battle_queues[player_id].put_nowait, battle
                        )
                        try:
                            self._logger.debug(
                                "[ENDSIG-PUT] %s ts=%.6f on_loop=False mode=scheduled",
                                player_id,
                                ts_put,
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass
                    try:
                        self._finished_enqueued[player_id].add(tag)
                    except Exception:
                        pass
                    try:
                        ts_set = time.monotonic()
                        POKE_LOOP.call_soon_threadsafe(self._finished_events[player_id].set)
                        try:
                            self._logger.debug(
                                "[ENDSIG-SET] %s ts=%.6f on_loop=False",
                                player_id,
                                ts_set,
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass
            
        except Exception:
            self._logger.exception("failed to notify battle finished for %s", player_id)
        else:
            # Record FD stats once per battle tag when a battle is marked finished
            try:
                if not hasattr(self, "_fd_bf_logged"):
                    self._fd_bf_logged: set[str] = set()
                tag = getattr(battle, "battle_tag", None)
                if isinstance(tag, str) and tag not in self._fd_bf_logged:
                    self._log_fd_stats("battle_finished_notify")
                    self._fd_bf_logged.add(tag)
            except Exception:
                # Fallback without dedup if anything goes wrong
                self._log_fd_stats("battle_finished_notify")

        # Also notify rQID waiters that this player's battle has been closed.
        # This prevents any in-flight wait_for_rqid_change from hanging past battle end.
        try:
            notifier = get_global_rqid_notifier()
            notifier.close_battle(player_id)
            try:
                self._logger.debug(
                    "[RQID-CLOSE] player=%s battle=%s closed_at=%.6f",
                    player_id,
                    getattr(battle, "battle_tag", None),
                    time.monotonic(),
                )
            except Exception:
                pass
        except Exception:
            # Per policy: no fallback, but don't crash finish signaling path
            self._logger.exception("rqid notifier close failed for %s", player_id)

    def _race_get(
        self,
        queue: asyncio.Queue[Any],
        *events: asyncio.Event,
    ) -> Any | None:
        """Return queue item or ``None`` if any event fires first."""

        async def _race() -> Any | None:
            # Identify pid for logging by matching queue object
            pid_label = None
            try:
                for _pid, _q in self._battle_queues.items():
                    if _q is queue:
                        pid_label = _pid
                        break
            except Exception:
                pass
            ts_start = time.monotonic()
            try:
                qsize0 = queue.qsize()
            except Exception:
                qsize0 = -1
            # By convention, the last event (if provided) is the finished_event
            finished_idx = len(events) - 1 if events else None
            
            get_task = asyncio.create_task(queue.get())
            wait_tasks = [asyncio.create_task(e.wait()) for e in events]
            done, pending = await asyncio.wait(
                {get_task, *wait_tasks},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for p in pending:
                p.cancel()
            if get_task in done:
                try:
                    dt_ms = int((time.monotonic() - ts_start) * 1000)
                    if pid_label is not None:
                        emit_metric(
                            "race_get_decision",
                            pid=pid_label,
                            decision="get_task",
                            qsize0=qsize0,
                            latency_ms=dt_ms,
                        )
                except Exception:
                    pass
                return get_task.result()
            # Some event fired first
            
            # If the finished_event fired, consume the final snapshot from the queue
            try:
                if finished_idx is not None and 0 <= finished_idx < len(wait_tasks) and wait_tasks[finished_idx] in done:
                    try:
                        dt_ms = int((time.monotonic() - ts_start) * 1000)
                        if pid_label is not None:
                            emit_metric(
                                "race_get_decision",
                                pid=pid_label,
                                decision="finished_event",
                                qsize0=qsize0,
                                latency_ms=dt_ms,
                            )
                    except Exception:
                        pass
                    # Block until the final snapshot is available (bounded by outer timeout)
                    fin_battle = await queue.get()
                    
                    return fin_battle
            except Exception:
                pass
            # イベントが先に完了した場合でも、キューにデータが残っていれば取得する
            if not queue.empty():
                try:
                    dt_ms = int((time.monotonic() - ts_start) * 1000)
                    if pid_label is not None:
                        emit_metric(
                            "race_get_decision",
                            pid=pid_label,
                            decision="drain_after_event",
                            qsize0=qsize0,
                            latency_ms=dt_ms,
                        )
                except Exception:
                    pass
                return await queue.get()
            # micro-sleep は撤廃。イベントに依存し、即時 None を返す。
            try:
                dt_ms = int((time.monotonic() - ts_start) * 1000)
                if pid_label is not None:
                    emit_metric(
                        "race_get_decision",
                        pid=pid_label,
                        decision="event_none",
                        qsize0=qsize0,
                        latency_ms=dt_ms,
                    )
            except Exception:
                pass
            return None

        
        try:
            result = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(_race(), self.timeout),
                POKE_LOOP,
            ).result()
        except Exception as exc:
            self._logger.error(
                "[TIMEOUT] race_get queue=%d events=%s exc=%s",
                queue.qsize(),
                [e.is_set() for e in events],
                exc,
            )
            raise
        
        if result is not None:
            POKE_LOOP.call_soon_threadsafe(queue.task_done)
        return result

    def _build_action_mask(
        self, action_mapping: dict[int, tuple[str, str | int, bool]]
    ) -> np.ndarray:
        """Return an action mask derived from ``action_mapping``."""

        return np.array(
            [
                0 if disabled else 1
                for _, (_, _, disabled) in sorted(action_mapping.items())
            ],
            dtype=np.int8,
        )

    def _compute_all_masks(self, battles: dict[str, Any] | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return current legal action masks for both players.

        When ``battles`` is provided, compute masks from this per-step snapshot
        rather than re-reading ``self._current_battles``.
        """

        masks: list[np.ndarray] = []
        for pid in self.agent_ids:
            battle = None
            if battles is not None:
                battle = battles.get(pid)
            if battle is None:
                battle = self.get_current_battle(pid)
            if battle is None:
                masks.append(np.zeros(self.ACTION_SIZE, dtype=np.int8))
                self._action_mappings[pid] = {}
                continue

            mapping = self.action_helper.get_action_mapping(battle)
            mask = self._build_action_mask(mapping)
            
            # Suppress noisy prints during mask computation; rely on explicit debug logs when needed
            # if all(m == 0 for m in mask):
            #     self._logger.debug(
            #         "All actions disabled for %s. force_switch=%s moves=%d switches=%d",
            #         pid,
            #         getattr(battle, 'force_switch', 'N/A'),
            #         len(getattr(battle, 'available_moves', [])),
            #         len(getattr(battle, 'available_switches', [])),
            #     )


            
            
            # Add additional debug info about the active Pokemon
            active_pokemon = getattr(battle, "active_pokemon", None)
                
            # Log all team members' active status
            


            # Team restriction system removed - all switches allowed based on battle state only


            self._action_mappings[pid] = mapping
            masks.append(mask)


        return tuple(masks)  # type: ignore[return-value]

    async def _process_actions_parallel(self, action_dict: dict[str, int | str]) -> None:
        """Process actions for all agents concurrently (Phase 1 optimization)."""
        
        async def _process_single_action(agent_id: str, action: int | str):
            """Process action for single agent asynchronously."""
            if isinstance(action, str):
                # Log context around string actions (e.g., team preview)
                try:
                    battle = self._current_battles.get(agent_id)
                    lr = getattr(battle, "last_request", None)
                    rq = lr.get("rqid") if isinstance(lr, dict) else None
                    tp = bool(lr.get("teamPreview")) if isinstance(lr, dict) else False
                    wt = bool(lr.get("wait")) if isinstance(lr, dict) else False
                    fs = bool(lr.get("forceSwitch")) if isinstance(lr, dict) else False
                    self._logger.debug(
                        "[ACTCTX] %s enqueue STR action=%r need_action=%s rqid=%s type=%s",
                        agent_id,
                        action,
                        self._need_action.get(agent_id, None),
                        rq,
                        ("teampreview" if tp else ("wait" if wt else ("force" if fs else "normal"))),
                    )
                except Exception:
                    pass
                await self._action_queues[agent_id].put(action)
                return

            # Enforce need_action gate for numeric actions based on current request type
            if not self._need_action.get(agent_id, True):
                # Drop silently but with debug log to avoid invalid sends (e.g., wait:true)
                try:
                    battle = self._current_battles.get(agent_id)
                    lr = getattr(battle, "last_request", None)
                    rq = lr.get("rqid") if isinstance(lr, dict) else None
                    tp = bool(lr.get("teamPreview")) if isinstance(lr, dict) else False
                    wt = bool(lr.get("wait")) if isinstance(lr, dict) else False
                    fs = bool(lr.get("forceSwitch")) if isinstance(lr, dict) else False
                    self._logger.debug(
                        "[ACTCTX] %s drop INT action=%r need_action=%s rqid=%s type=%s",
                        agent_id,
                        action,
                        self._need_action.get(agent_id, None),
                        rq,
                        ("teampreview" if tp else ("wait" if wt else ("force" if fs else "normal"))),
                    )
                except Exception:
                    pass
                return

            # Integer action processing
            if self.defer_action_conversion:
                # Defer int->BattleOrder conversion to EnvPlayer.choose_move
                # Log context about action, previous mask, and current request type for diagnosis
                try:
                    battle = self._current_battles.get(agent_id)
                    lr = getattr(battle, "last_request", None)
                    prev_mapping = self._action_mappings.get(agent_id, {})
                    prev_mask = self._build_action_mask(prev_mapping) if prev_mapping else None
                    was_enabled_prev = None
                    if prev_mask is not None and 0 <= int(action) < len(prev_mask):
                        was_enabled_prev = bool(prev_mask[int(action)])
                    rq = lr.get("rqid") if isinstance(lr, dict) else None
                    tp = bool(lr.get("teamPreview")) if isinstance(lr, dict) else False
                    wt = bool(lr.get("wait")) if isinstance(lr, dict) else False
                    fs = bool(lr.get("forceSwitch")) if isinstance(lr, dict) else False
                    # Validate rqid consistency: numeric action must target last mask's rqid
                    last_mask_rqid = self._mask_rqids.get(agent_id)
                    if last_mask_rqid is not None and rq is not None and rq != last_mask_rqid:
                        self._logger.debug(
                            "[ACTCTX] %s drop INT action=%s due to rqid mismatch current=%s expected(mask)=%s",
                            agent_id,
                            action,
                            rq,
                            last_mask_rqid,
                        )
                        return
                    self._logger.debug(
                        "[ACTCTX] %s enqueue INT action=%s need_action=%s prev_enabled=%s rqid=%s type=%s",
                        agent_id,
                        action,
                        self._need_action.get(agent_id, None),
                        was_enabled_prev,
                        rq,
                        ("teampreview" if tp else ("wait" if wt else ("force" if fs else "normal"))),
                    )
                except Exception:
                    pass
                await self._action_queues[agent_id].put(int(action))
                return

            # Legacy path: convert immediately using current battle snapshot
            battle = self._current_battles.get(agent_id)
            if battle is None:
                raise ValueError(f"No current battle for {agent_id}")

            mapping = self.action_helper.get_action_mapping(battle)
            self._action_mappings[agent_id] = mapping
            DisabledErr = getattr(self.action_helper, "DisabledMoveError", ValueError)
            try:
                order = self.action_helper.action_index_to_order_from_mapping(
                    self._env_players[agent_id], battle, int(action), mapping
                )
            except DisabledErr as e:
                err_msg = f"invalid action: {agent_id} selected {action} with mapping {mapping}"
                self._logger.error(err_msg)
                raise RuntimeError(err_msg) from e
            await self._action_queues[agent_id].put(order)
        
        # Validate all required actions are present
        for agent_id in self.agent_ids:
            if agent_id not in action_dict:
                raise ValueError(f"{agent_id} action required")
        
        # Process all actions concurrently
        tasks = []
        for agent_id in self.agent_ids:
            if agent_id in action_dict:
                task = _process_single_action(agent_id, action_dict[agent_id])
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks)


    def step(self, action_dict: dict[str, int | str], *, return_masks: bool = True):
        """マルチエージェント形式で1ステップ進める。

        ``return_masks`` を ``True`` にすると、戻り値の末尾に各プレイヤーの
        行動マスクを含むタプルを追加で返す。
        
        Phase 1 Optimization: Action processing is now parallelized for improved performance.
        """

        # Phase 1: Set need_action flags based on current request types (before sending actions)
        
        for pid in self.agent_ids:
            battle = self._current_battles.get(pid)
            lr = getattr(battle, "last_request", None)
            req_type = "none"
            if isinstance(lr, dict):
                if lr.get("teamPreview"):
                    req_type = "teampreview"
                elif lr.get("forceSwitch"):
                    req_type = "force"
                elif lr.get("wait"):
                    req_type = "wait"
                else:
                    req_type = "normal"
            # need_action: normal or force requires actions, wait/teampreview do not here
            self._need_action[pid] = req_type in ("normal", "force")

        # Phase 2: Parallel action processing
        asyncio.run_coroutine_threadsafe(
            self._process_actions_parallel(action_dict), POKE_LOOP
        ).result()

        # Check battle task status before waiting for new states
        if hasattr(self, "_battle_task") and self._battle_task.done():
            exc = self._battle_task.exception()
            if exc is not None:
                raise exc

        # Phase 2 (Future optimization): Battle state retrieval - currently sequential
        # TODO: Implement _retrieve_battles_parallel() for further performance gains
        battles: dict[str, Any] = {}
        # このステップで各プレイヤーの request/rqid が更新されたかを rqid ベースで検知する
        updated: dict[str, bool] = {}
        # Snapshot previous rqids for synchronization check
        prev_rqids: dict[str, int | None] = {pid: self._last_rqids.get(pid) for pid in self.agent_ids}
        for pid in self.agent_ids:
            opp = "player_1" if pid == "player_0" else "player_0"
            # Try to retrieve the latest battle update for this player
            
            # Always include finished_event in race; _race_get consumes from queue
            # when finished_event wins to prevent leftover snapshots.
            battle = self._race_get(
                self._battle_queues[pid],
                self._env_players[pid]._waiting,
                self._env_players[opp]._trying_again,
                self._finished_events[pid],
            )
            
            self._env_players[pid]._waiting.clear()
            if battle is None:
                self._env_players[opp]._trying_again.clear()
                battle = self._current_battles[pid]
                # Update need_action based on current request for next step
                lr = getattr(battle, "last_request", None)
                if isinstance(lr, dict):
                    if lr.get("teamPreview"):
                        self._need_action[pid] = False
                    elif lr.get("wait"):
                        self._need_action[pid] = False
                    elif lr.get("forceSwitch"):
                        self._need_action[pid] = True
                    else:
                        self._need_action[pid] = True
                else:
                    self._need_action[pid] = False
            else:
                self._current_battles[pid] = battle
                # Update need_action based on new request for next step
                lr = getattr(battle, "last_request", None)
                if isinstance(lr, dict):
                    if lr.get("teamPreview"):
                        self._need_action[pid] = False
                    elif lr.get("wait"):
                        self._need_action[pid] = False
                    elif lr.get("forceSwitch"):
                        self._need_action[pid] = True
                    else:
                        self._need_action[pid] = True
                else:
                    self._need_action[pid] = False
            battles[pid] = battle

            # rqid の差分、または update:true の付与を「更新」とみなす
            try:
                prev_lr = self._last_requests.get(pid)
                prev_rqid = (
                    prev_lr.get("rqid") if isinstance(prev_lr, dict) else None
                )
                curr_lr = getattr(battle, "last_request", None)
                curr_rqid = (
                    curr_lr.get("rqid") if isinstance(curr_lr, dict) else None
                )
                has_update_flag = bool(
                    isinstance(curr_lr, dict) and curr_lr.get("update") is True
                )
                updated[pid] = bool(
                    (prev_rqid is not None and curr_rqid is not None and curr_rqid != prev_rqid)
                    or has_update_flag
                )
            except Exception:
                updated[pid] = False
            # Snapshot per-pid summary after retrieval path is decided
            
            # Log current battle object and last_request summary for each player
            try:
                lr = getattr(battle, "last_request", None)
                rq = lr.get("rqid") if isinstance(lr, dict) else None
                tp = bool(lr.get("teamPreview")) if isinstance(lr, dict) else False
                wt = bool(lr.get("wait")) if isinstance(lr, dict) else False
                fs = bool(lr.get("forceSwitch")) if isinstance(lr, dict) else False
                self._logger.debug(
                    "[RQSRC] %s obj=%s tag=%s lr_type=%s rqid=%s wait=%s force=%s tp=%s prev_rqid=%s",
                    pid,
                    hex(id(battle)) if battle else None,
                    getattr(battle, "battle_tag", "?"),
                    ("teampreview" if tp else ("wait" if wt else ("force" if fs else "normal"))),
                    rq,
                    wt,
                    fs,
                    tp,
                    prev_rqids.get(pid),
                )
            except Exception:
                pass

        # RQID synchronization before building observations (also during teampreview)
        def _get_rqid(b: Any) -> int | None:
            lr = getattr(b, "last_request", None)
            return lr.get("rqid") if isinstance(lr, dict) else None

        # 対向プレイヤーの rqid 変化状況を参照できるように現在 rqid を収集
        current_rqids: dict[str, int | None] = {pid: _get_rqid(battles[pid]) for pid in self.agent_ids}

        def _needs_update(pid: str, b: Any, prev: int | None) -> bool:
            """片側だけの rqid 更新を許容し、必要な側のみ待機する判定。

            - finished: 待たない
            - teamPreview: 従来通り待つ
            - update:true: 直ちに処理（待たない）
            - このステップで更新が検知されていない側は待たない
            - それ以外は、前回 rqid が存在し、かつ未変化のときのみ待つ
            - ただし相手側がこのステップで rqid 更新（または update:true）している場合は待たない
            """
            if getattr(b, "finished", False):
                return False
            lr = getattr(b, "last_request", None)
            if not isinstance(lr, dict):
                # 構造化 request が未到着なら一旦待つ
                return True

            if lr.get("teamPreview"):
                return True

            if lr.get("update") is True:
                return False

            # このステップで自身に更新がなければ待たない
            if not updated.get(pid, False):
                return False

            # 相手がこのステップで更新しているなら、自身の未変化は待たない
            opp = "player_1" if pid == "player_0" else "player_0"
            if updated.get(opp, False):
                return False

            # 上記いずれにも該当しなければ rqid の未変化を待つ
            if prev is not None:
                curr = lr.get("rqid")
                return curr == prev
            return False

        pending: set[str] = set(
            pid for pid in self.agent_ids if _needs_update(pid, battles[pid], prev_rqids.get(pid))
        )

        # RQID 同期: 以降は同一 Battle オブジェクトの in-place 更新を待つ。
        # 追加のバトル更新キューのドレインは行わず、PSClient による
        # battle.last_request の更新が反映されるまで短いスリープで再評価する。
        if pending:
            rqid_sync_start = time.monotonic()
            iterations = 0  # kept for metrics compatibility
            deadline = time.monotonic() + float(self.timeout)

            # Build notifier and per-pid wait futures lazily
            notifier = get_global_rqid_notifier()
            wait_futs: dict[str, cf.Future] = {}

            def _ensure_future(pid: str) -> None:
                if pid in wait_futs and not wait_futs[pid].done():
                    return
                baseline = prev_rqids.get(pid)
                coro = notifier.wait_for_rqid_change(pid, baseline_rqid=baseline, timeout=max(0.0, deadline - time.monotonic()))
                # Schedule on POKE_LOOP to match notifier loop usage
                wait_futs[pid] = asyncio.run_coroutine_threadsafe(coro, POKE_LOOP)

            # Initial futures for current pending set
            for pid in list(pending):
                # Double-check need before registering waits
                if _needs_update(pid, battles[pid], prev_rqids.get(pid)):
                    _ensure_future(pid)
                else:
                    pending.discard(pid)

            # Event-driven loop: wait for any rqid change, then re-evaluate
            while pending and time.monotonic() < deadline:
                # Filter current valid futures
                current_futs = [wait_futs[pid] for pid in pending if pid in wait_futs]
                if not current_futs:
                    # No active waits; create them for remaining pids
                    for pid in list(pending):
                        _ensure_future(pid)
                    current_futs = [wait_futs[pid] for pid in pending if pid in wait_futs]
                    if not current_futs:
                        break

                remaining = max(0.0, deadline - time.monotonic())
                done, not_done = cf.wait(current_futs, timeout=remaining, return_when=cf.FIRST_COMPLETED)

                # Any completion (success or exception) should trigger re-check
                if not done:
                    # Overall timeout reached
                    break

                # Consume results to surface exceptions for diagnostics but continue loop
                for fut in done:
                    try:
                        _ = fut.result()
                    except Exception:
                        # Errors are handled by timeout/error path later
                        pass

                # Re-evaluate all pids against current battles state
                for pid in list(pending):
                    b = battles[pid]
                    if not _needs_update(pid, b, prev_rqids.get(pid)):
                        pending.discard(pid)
                        # Cancel and drop its future if any
                        fut = wait_futs.pop(pid, None)
                        if fut is not None and not fut.done():
                            try:
                                fut.cancel()
                            except Exception:
                                pass
                    else:
                        # Ensure a new wait is registered if the previous one completed
                        _ensure_future(pid)

        # Log success metrics when pending cleared without timeout
        if not pending:
            try:
                dt_ms = int((time.monotonic() - rqid_sync_start) * 1000) if 'rqid_sync_start' in locals() else 0
                emit_metric(
                    "rqid_sync_success",
                    rqid_wait_latency_ms=dt_ms,
                    rqid_poll_iterations=(iterations if 'iterations' in locals() else 0),
                )
            except Exception:
                pass
            # FD snapshot on successful rqid sync
            try:
                self._log_fd_stats("rqid_sync_success")
            except Exception:
                pass

        if pending:
            # Log detailed error per player then exit fatally
            for pid in sorted(pending):
                b = battles.get(pid, None) or self._current_battles.get(pid)
                curr = _get_rqid(b) if b is not None else None
                prev = prev_rqids.get(pid)
                sd_id = "p1" if pid == "player_0" else "p2"
                lr = getattr(b, "last_request", None) if b is not None else None
                lr_type = (
                    "teampreview" if isinstance(lr, dict) and lr.get("teamPreview") else ("normal" if isinstance(lr, dict) else "none")
                )
                try:
                    dt_ms = int((time.monotonic() - rqid_sync_start) * 1000) if 'rqid_sync_start' in locals() else 0
                    emit_metric(
                        "rqid_sync_timeout",
                        pid=pid,
                        rqid_timeouts_count=1,
                        elapsed_ms=dt_ms,
                        iterations=(iterations if 'iterations' in locals() else 0),
                        prev_rqid=prev,
                        curr_rqid=curr,
                        lr_type=lr_type,
                    )
                except Exception:
                    pass
                self._logger.error(
                    "[RQID TIMEOUT] %s(%s) battle=%s turn=%s teampreview=%s prev_rqid=%s curr_rqid=%s last_request_type=%s obj=%s",
                    pid,
                    sd_id,
                    getattr(b, "battle_tag", "?"),
                    getattr(b, "turn", "?"),
                    self._is_teampreview(b) if b is not None else "?",
                    prev,
                    curr,
                    lr_type,
                    hex(id(b)) if b else None,
                )
            raise SystemExit(1)

        # Update last rqids after successful sync
        for pid in self.agent_ids:
            self._last_rqids[pid] = _get_rqid(battles[pid])

        # Build masks from this step snapshot if enabled
        if self.use_snapshot_masks:
            masks = self._compute_all_masks(battles)
        else:
            masks = self._compute_all_masks()
        for pid in self.agent_ids:
            self._last_requests[pid] = self._current_battles[pid].last_request
            # Also track rqid used for these masks
            lr = getattr(self._current_battles[pid], "last_request", None)
            self._mask_rqids[pid] = (lr.get("rqid") if isinstance(lr, dict) else None)

        # Check if we're in teampreview phase for step() as well
        battle0 = battles["player_0"]
        if self._is_teampreview(battle0):
            self._logger.debug("In teampreview phase during step() - using dummy observations")
            observations = {
                "player_0": self._get_teampreview_observation(),
                "player_1": self._get_teampreview_observation(),
            }
        else:
            observations = {
                pid: self.state_observer.observe(battles[pid]) for pid in self.agent_ids
            }
                
        rewards = {pid: self._calc_reward(battles[pid], pid) for pid in self.agent_ids}
        
        # Reset invalid action flags after reward calculation
        for pid in self.agent_ids:
            battle = battles[pid]
            if hasattr(battle, 'reset_invalid_action'):
                battle.reset_invalid_action()
        terminated: dict[str, bool] = {}
        truncated: dict[str, bool] = {}
        for pid in self.agent_ids:
            term, trunc = self._check_episode_end(battles[pid])
            terminated[pid] = term
            truncated[pid] = trunc
        if any(truncated.values()):
            rewards = {agent_id: 0.0 for agent_id in self.agent_ids}

        infos = {agent_id: {} for agent_id in self.agent_ids}

        # Drain any leftover final snapshots to avoid queue.join() blocking and stalls.
        try:
            for pid in self.agent_ids:
                b = battles.get(pid)
                is_finished = bool(getattr(b, "finished", False))
                if not is_finished:
                    continue
                # Only drain if the finished event is set and queue has remaining items
                q = self._battle_queues.get(pid)
                ev = self._finished_events.get(pid)
                if q is None or ev is None or not ev.is_set():
                    continue

                async def _drain(q: asyncio.Queue[Any]) -> int:
                    drained = 0
                    while True:
                        try:
                            item = q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        else:
                            drained += 1
                            q.task_done()
                    return drained

                prev_qsize = q.qsize()
                drained = asyncio.run_coroutine_threadsafe(_drain(q), POKE_LOOP).result()
                
        except Exception:
            # Draining is a best-effort safeguard; failures should not break the step
            pass

        if hasattr(self, "single_agent_mode"):
            mask0 = masks[0]
            done = terminated[self.agent_ids[0]] or truncated[self.agent_ids[0]]
            if return_masks:
                return (
                    observations[self.agent_ids[0]],
                    mask0,
                    rewards[self.agent_ids[0]],
                    done,
                    infos[self.agent_ids[0]],
                )
            return (
                observations[self.agent_ids[0]],
                rewards[self.agent_ids[0]],
                done,
                infos[self.agent_ids[0]],
            )

        if return_masks:
            return observations, rewards, terminated, truncated, infos, masks
        return observations, rewards, terminated, truncated, infos

    # Step13: 終了判定ユーティリティ
    def _check_episode_end(self, battle: Any) -> tuple[bool, bool]:
        """Return ``(terminated, truncated)`` for ``battle``."""

        terminated = bool(getattr(battle, "finished", False))
        truncated = getattr(battle, "turn", 0) > self.MAX_TURNS
        return terminated, truncated

    # Step11: 報酬計算ユーティリティ
    def _calc_reward(self, battle: Any, pid: str) -> float:
        """HP差報酬に勝敗ボーナスを加算して返す。"""

        hp_reward = 0.0
        raw_reward = 0.0
        
        if self.reward_type == "hp_delta" and pid in self._hp_delta_rewards:
            hp_reward = self._hp_delta_rewards[pid].calc(battle)

            win_reward = 0.0
            if getattr(battle, "finished", False):
                win_reward = 10.0 if getattr(battle, "won", False) else -10.0

            raw_reward = float(hp_reward + win_reward)

        elif self.reward_type == "composite" and pid in self._composite_rewards:
            total = self._composite_rewards[pid].calc(battle)
            self._sub_reward_logs[pid] = dict(
                self._composite_rewards[pid].last_values
            )
            
            # Add win/loss reward to the breakdown
            win_reward = 0.0
            if getattr(battle, "finished", False):
                win_reward = 10.0 if getattr(battle, "won", False) else -10.0
                self._sub_reward_logs[pid]["win_loss"] = win_reward
            
            raw_reward = float(total + win_reward)
        else:
            win_reward = 0.0
            if getattr(battle, "finished", False):
                win_reward = 10.0 if getattr(battle, "won", False) else -10.0
            raw_reward = float(hp_reward + win_reward)
        
        # Apply reward normalization if enabled
        if self.normalize_rewards and pid in self._reward_normalizers:
            self._reward_normalizers[pid].update(raw_reward)
            normalized_reward = self._reward_normalizers[pid].normalize(raw_reward)
            return float(normalized_reward)
        
        return raw_reward

    def get_reward_normalization_stats(self) -> dict[str, dict[str, float]]:
        """Get reward normalization statistics for all agents.
        
        Returns:
            Dictionary mapping agent IDs to their normalization stats
        """
        stats = {}
        if self.normalize_rewards:
            for agent_id, normalizer in self._reward_normalizers.items():
                stats[agent_id] = normalizer.get_stats()
        return stats

    def render(self) -> None:
        """Render the current battle state to the console."""

        battle = getattr(self, "_current_battles", {}).get("player_0")
        if battle is None:
            return None

        try:
            team_left = "".join("⦻" if m.fainted else "●" for m in battle.team.values())
            team_right = "".join(
                "⦻" if m.fainted else "●" for m in battle.opponent_team.values()
            )
            active = battle.active_pokemon
            opp = battle.opponent_active_pokemon
            active_hp = active.current_hp or 0
            active_max = active.max_hp or 0
            opp_hp = opp.current_hp or 0
            log_line = (
                f"  Turn {battle.turn:4d}. | [{team_left}][{active_hp:3d}/{active_max:3d}hp] "
                f"{active.species:10.10s} - {opp.species:10.10s} [{opp_hp:3d}hp][{team_right}]"
            )
            self._logger.info(log_line)
            print(log_line, end="\n" if battle.finished else "\r")
        except Exception as exc:  # pragma: no cover - render failures shouldn't break
            self._logger.debug("render error: %s", exc)
        return None

    def close(self) -> None:
        """Terminate ongoing tasks and close WebSocket connections."""

        if hasattr(self, "_battle_task"):
            self._battle_task.cancel()
            try:
                self._battle_task.result()
            except Exception:
                pass
            del self._battle_task

        if hasattr(self, "_env_players"):
            for p in self._env_players.values():
                if self.battle_mode == "online":
                    # Try player's own close_connection first
                    try:
                        if hasattr(p, "close_connection"):
                            asyncio.run_coroutine_threadsafe(
                                p.close_connection(), POKE_LOOP
                            ).result()
                    except Exception as e:
                        self._logger.exception(
                            "[WS_CLEANUP] player.close_connection failed pid=%s: %s",
                            getattr(p, "player_id", "unknown"),
                            e,
                        )
                        raise
                    # Force-close only if websocket remains
                    try:
                        ps_client = getattr(p, "ps_client", None)
                        if ps_client is not None and getattr(ps_client, "websocket", None) is not None:
                            self._force_close_psclient(ps_client, getattr(p, "player_id", None))
                    except Exception as e:
                        self._logger.exception(
                            "[WS_CLEANUP] force_close_psclient failed pid=%s: %s",
                            getattr(p, "player_id", "unknown"),
                            e,
                        )
                        raise
            self._env_players.clear()

        # Drain any leftover items from queues to avoid join() blocking
        async def _drain(q: asyncio.Queue[Any]) -> int:
            drained = 0
            while True:
                try:
                    item = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    drained += 1
                    q.task_done()
            return drained

        for q in list(self._action_queues.values()):
            try:
                asyncio.run_coroutine_threadsafe(_drain(q), POKE_LOOP).result()
            except Exception:
                pass
            asyncio.run_coroutine_threadsafe(q.join(), POKE_LOOP).result()

        for q in list(self._battle_queues.values()):
            try:
                asyncio.run_coroutine_threadsafe(_drain(q), POKE_LOOP).result()
            except Exception:
                pass
            asyncio.run_coroutine_threadsafe(q.join(), POKE_LOOP).result()

        # Clear finished events to release any waiters
        try:
            for ev in getattr(self, "_finished_events", {}).values():
                try:
                    ev.clear()
                except Exception:
                    pass
        except Exception:
            pass
        self._action_queues.clear()
        self._battle_queues.clear()
        # FD snapshot after environment close
        self._log_fd_stats("env_close")

    def _validate_battle_mode_config(self, config: dict) -> None:
        """Validate battle mode configuration."""
        try:
            validate_mode_configuration(self.battle_mode, config)
            self._logger.info(f"Battle mode '{self.battle_mode}' configuration validated")
        except ValueError as e:
            self._logger.error(f"Invalid battle mode configuration: {e}")
            raise

    def _create_battle_player(self, player_id: str, server_config: Any, team: Any, account_config: Any) -> Any:
        """Create a battle player based on the current battle mode.
        
        Args:
            player_id: Player identifier ("player_0" or "player_1")
            server_config: Server configuration object
            team: Team configuration for the player
            account_config: Account configuration for the player
            
        Returns:
            Player instance (DualModeEnvPlayer or EnvPlayer based on mode)
        """
        if self.battle_mode == "local":
            self._logger.info(f"Creating local IPC player: {player_id}")
            return DualModeEnvPlayer(
                env=self,
                player_id=player_id,
                mode="local",
                server_configuration=server_config,  # Pass server config for fallback
                battle_format="gen9bssregi",
                team=team,
                log_level=self.log_level,
                save_replays=self.save_replays,
                account_configuration=account_config,
                reuse_processes=self.reuse_processes,
                max_processes=self.max_processes,
            )
        else:
            self._logger.info(f"Creating online WebSocket player: {player_id}")
            return DualModeEnvPlayer(
                env=self,
                player_id=player_id,
                mode="online",
                server_configuration=server_config,
                battle_format="gen9bssregi",
                team=team,
                log_level=self.log_level,
                save_replays=self.save_replays,
                account_configuration=account_config,
            )
    
    def get_battle_mode(self) -> str:
        """Get current battle mode.
        
        Returns:
            Current battle mode ("local" or "online")
        """
        return self.battle_mode
    
    def set_battle_mode(self, mode: str) -> None:
        """Set battle mode (affects new battles only).
        
        Args:
            mode: Battle mode ("local" or "online")
            
        Raises:
            ValueError: If mode is not supported
        """
        if mode not in ["local", "online"]:
            raise ValueError(f"Unsupported battle mode: {mode}. Use 'local' or 'online'")
        
        old_mode = self.battle_mode
        self.battle_mode = mode
        self._logger.info(f"Battle mode changed from '{old_mode}' to '{mode}'")
        
    def get_battle_mode_info(self) -> dict:
        """Get information about the current battle mode.
        
        Returns:
            Dictionary containing battle mode information
        """
        return {
            "current_mode": self.battle_mode,
            "supported_modes": ["local", "online"],
            "mode_descriptions": {
                "local": "High-speed IPC communication with embedded Node.js process",
                "online": "Traditional WebSocket communication with Pokemon Showdown servers"
            },
            "current_server_config": getattr(self, "server_configuration", None),
            "players_created": hasattr(self, "_env_players") and bool(self._env_players)
        }
    
    # ------------------------------------------------------------------
    # Battle State Management (Phase 3)
    # ------------------------------------------------------------------
    
    def save_battle_state(self, agent_id: str = "player_0", filename: str | None = None) -> str:
        """Save current battle state to file.
        
        Args:
            agent_id: Agent whose battle to save
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved state file
            
        Raises:
            ValueError: If no current battle for agent_id
            RuntimeError: If serialization fails
        """
        try:
            battle = self.get_current_battle(agent_id)
            if battle is None:
                raise ValueError(f"No current battle for agent {agent_id}")
            
            filepath = self._state_manager.save_state(battle, filename)
            self._logger.info(f"Saved battle state for {agent_id} to {filepath}")
            return filepath
            
        except Exception as e:
            self._logger.error(f"Failed to save battle state: {e}")
            raise RuntimeError(f"Battle state save failed: {e}") from e
    
    def load_battle_state(self, filepath: str) -> dict:
        """Load battle state from file.
        
        Args:
            filepath: Path to saved state file
            
        Returns:
            Dictionary containing the loaded battle state
            
        Raises:
            FileNotFoundError: If state file not found
            RuntimeError: If deserialization fails
        """
        try:
            state = self._state_manager.load_state(filepath)
            self._logger.info(f"Loaded battle state from {filepath}")
            return state.to_dict()
            
        except FileNotFoundError:
            self._logger.error(f"Battle state file not found: {filepath}")
            raise
        except Exception as e:
            self._logger.error(f"Failed to load battle state: {e}")
            raise RuntimeError(f"Battle state load failed: {e}") from e
    
    def list_saved_battle_states(self, battle_id: str | None = None) -> list[str]:
        """List available saved battle states.
        
        Args:
            battle_id: Optional filter by battle ID
            
        Returns:
            List of available state files
        """
        try:
            states = self._state_manager.list_saved_states(battle_id)
            self._logger.debug(f"Found {len(states)} saved battle states")
            return states
            
        except Exception as e:
            self._logger.error(f"Failed to list saved states: {e}")
            return []
    
    def delete_battle_state(self, filename: str) -> bool:
        """Delete a saved battle state file.
        
        Args:
            filename: Name of file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self._state_manager.delete_state(filename)
            if success:
                self._logger.info(f"Deleted battle state: {filename}")
            else:
                self._logger.warning(f"Battle state not found: {filename}")
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to delete battle state: {e}")
            return False
    
    def get_battle_state_info(self) -> dict:
        """Get information about battle state management.
        
        Returns:
            Dictionary containing state management information
        """
        try:
            saved_states = self.list_saved_battle_states()
            current_battles = []
            
            # Check current battles
            for agent_id in self.agent_ids:
                battle = self.get_current_battle(agent_id)
                if battle:
                    current_battles.append({
                        "agent_id": agent_id,
                        "battle_id": getattr(battle, 'battle_tag', 'unknown'),
                        "turn": getattr(battle, 'turn', 0),
                        "finished": getattr(battle, 'finished', False)
                    })
            
            return {
                "serializer_type": type(self._battle_serializer).__name__,
                "storage_directory": str(self._state_manager.storage_dir),
                "saved_states_count": len(saved_states),
                "saved_states": saved_states[:10],  # Show first 10
                "current_battles": current_battles,
                "battle_mode": self.battle_mode
            }
            
        except Exception as e:
            self._logger.error(f"Failed to get state info: {e}")
            return {
                "error": str(e),
                "serializer_type": type(self._battle_serializer).__name__,
                "storage_directory": str(self._state_manager.storage_dir)
            }
    
    async def save_battle_state_via_communicator(self, agent_id: str = "player_0") -> dict:
        """Save battle state via communicator (for dual-mode support).
        
        This method uses the communicator's save_battle_state method for
        mode-specific optimizations (e.g., direct IPC state saving).
        
        Args:
            agent_id: Agent whose battle to save
            
        Returns:
            Dictionary containing save operation result
            
        Raises:
            ValueError: If agent has no communicator or no current battle
            RuntimeError: If save operation fails
        """
        try:
            # Get player and battle for agent
            player = getattr(self, '_env_players', {}).get(agent_id)
            if player is None:
                raise ValueError(f"No player found for agent {agent_id}")
            
            battle = self.get_current_battle(agent_id)
            if battle is None:
                raise ValueError(f"No current battle for agent {agent_id}")
            
            # Check if player has communicator (dual-mode player)
            if hasattr(player, '_communicator') and player._communicator:
                battle_id = getattr(battle, 'battle_tag', f'battle_{agent_id}')
                result = await player._communicator.save_battle_state(battle_id)
                self._logger.info(f"Saved battle state via communicator for {agent_id}")
                return result
            else:
                # Fallback to local serialization
                self._logger.info(f"Using local serialization for {agent_id} (no communicator)")
                filepath = self.save_battle_state(agent_id)
                return {
                    "type": "battle_state_saved",
                    "battle_id": getattr(battle, 'battle_tag', f'battle_{agent_id}'),
                    "method": "local_file",
                    "filepath": filepath,
                    "success": True
                }
                
        except Exception as e:
            self._logger.error(f"Failed to save battle state via communicator: {e}")
            raise RuntimeError(f"Communicator state save failed: {e}") from e
    
    async def restore_battle_state_via_communicator(
        self, 
        agent_id: str, 
        state_data: dict
    ) -> bool:
        """Restore battle state via communicator (for dual-mode support).
        
        Args:
            agent_id: Agent whose battle to restore
            state_data: Previously saved state data
            
        Returns:
            True if restoration was successful
            
        Raises:
            ValueError: If agent has no communicator
            RuntimeError: If restore operation fails
        """
        try:
            # Get player for agent
            player = getattr(self, '_env_players', {}).get(agent_id)
            if player is None:
                raise ValueError(f"No player found for agent {agent_id}")
            
            # Check if player has communicator (dual-mode player)
            if hasattr(player, '_communicator') and player._communicator:
                battle_id = state_data.get('battle_id', f'battle_{agent_id}')
                success = await player._communicator.restore_battle_state(battle_id, state_data)
                
                if success:
                    self._logger.info(f"Restored battle state via communicator for {agent_id}")
                else:
                    self._logger.error(f"Failed to restore battle state via communicator for {agent_id}")
                
                return success
            else:
                self._logger.warning(f"No communicator available for {agent_id} - cannot restore via communicator")
                return False
                
        except Exception as e:
            self._logger.error(f"Failed to restore battle state via communicator: {e}")
            raise RuntimeError(f"Communicator state restore failed: {e}") from e
    
