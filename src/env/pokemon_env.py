"""Gymnasium environment skeleton for Pokémon battles."""

from __future__ import annotations

from typing import Any, Tuple
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
                    if hasattr(p, "close"):
                        try:
                            p.close()
                        except Exception:
                            pass  # Ignore cleanup errors
            
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
            # Update current battle reference
            self._current_battles[player_id] = battle
            # Signal finished event on the asyncio loop thread-safely
            POKE_LOOP.call_soon_threadsafe(self._finished_events[player_id].set)
            # Also enqueue the latest battle snapshot for consumers waiting on queue
            asyncio.run_coroutine_threadsafe(
                self._battle_queues[player_id].put(battle), POKE_LOOP
            )
            try:
                self._logger.debug(
                    "[ENDSIG] %s notify finished tag=%s obj=%s qsize=%d",
                    player_id,
                    getattr(battle, "battle_tag", None),
                    hex(id(battle)),
                    self._battle_queues[player_id].qsize(),
                )
            except Exception:
                pass
        except Exception:
            self._logger.exception("failed to notify battle finished for %s", player_id)

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
                self._logger.debug(
                    "[RACE] start pid=%s ts=%.6f qsize=%d events=%s",
                    pid_label,
                    ts_start,
                    queue.qsize(),
                    [e.is_set() for e in events],
                )
            except Exception:
                pass
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
                    self._logger.debug(
                        "[RACE] done pid=%s kind=get elapsed_ms=%.1f rem_qsize=%d",
                        pid_label,
                        (time.monotonic() - ts_start) * 1000.0,
                        queue.qsize(),
                    )
                except Exception:
                    pass
                return get_task.result()
            # Some event fired first
            try:
                fired = [i for i, t in enumerate(wait_tasks) if t in done]
                self._logger.debug(
                    "[RACE] event pid=%s idx=%s elapsed_ms=%.1f qsize=%d",
                    pid_label,
                    fired,
                    (time.monotonic() - ts_start) * 1000.0,
                    queue.qsize(),
                )
            except Exception:
                pass
            # イベントが先に完了した場合でも、キューにデータが残っていれば取得する
            if not queue.empty():
                return await queue.get()
            # ---- 遅延対策 -------------------------------------------------
            # _waiting イベントがトリガーされた直後にキューへバトルデータが
            # 追加される場合がある。直後にチェックすると空のままになってしまい
            # マスク生成に失敗するため、僅かに待機してから再確認する。
            for _ in range(10):
                await asyncio.sleep(0.05)  # タスク切り替えを促す
                if not queue.empty():
                    return await queue.get()
            return None

        self._logger.debug(
            "[DBG] race_get start qsize=%d events=%s",
            queue.qsize(),
            [e.is_set() for e in events],
        )
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
        self._logger.debug(
            "[DBG] race_get done result=%s qsize=%d", result, queue.qsize()
        )
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


            switches_info = [
                f"{getattr(p, 'species', '?')}"
                f"(HP:{getattr(p, 'current_hp_fraction', 0) * 100:.1f}%"
                f", fainted={getattr(p, 'fainted', False)}"
                f", active={getattr(p, 'active', False)})"
                for p in getattr(battle, "available_switches", [])
            ]
            self._logger.debug("[DBG] %s available_switches: %s", pid, switches_info)
            
            # Add additional debug info about the active Pokemon
            active_pokemon = getattr(battle, "active_pokemon", None)
            if active_pokemon:
                self._logger.debug(
                    "[DBG] %s active_pokemon: species=%s, active=%s, ident=%s",
                    pid,
                    getattr(active_pokemon, 'species', '?'),
                    getattr(active_pokemon, 'active', '?'),
                    getattr(active_pokemon, '_ident', '?')
                )
                
            # Log all team members' active status
            team_info = []
            for poke_id, poke in getattr(battle, "team", {}).items():
                team_info.append(
                    f"{poke_id}: {getattr(poke, 'species', '?')} "
                    f"(active={getattr(poke, 'active', '?')})"
                )
            self._logger.debug("[DBG] %s team status: %s", pid, team_info)


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
        try:
            ts = time.monotonic()
            for _pid in self.agent_ids:
                _b = getattr(self, "_current_battles", {}).get(_pid)
                lr = getattr(_b, "last_request", None) if _b is not None else None
                rq = (lr.get("rqid") if isinstance(lr, dict) else None)
                tp = (bool(lr.get("teamPreview")) if isinstance(lr, dict) else None)
                wt = (bool(lr.get("wait")) if isinstance(lr, dict) else None)
                fs = (bool(lr.get("forceSwitch")) if isinstance(lr, dict) else None)
                self._logger.debug(
                    "[STEP] begin ts=%.6f pid=%s finished=%s turn=%s obj=%s tag=%s rqid=%s tp=%s wt=%s fs=%s",
                    ts,
                    _pid,
                    (getattr(_b, "finished", None) if _b is not None else None),
                    (getattr(_b, "turn", None) if _b is not None else None),
                    (hex(id(_b)) if _b is not None else None),
                    (getattr(_b, "battle_tag", None) if _b is not None else None),
                    rq,
                    tp,
                    wt,
                    fs,
                )
        except Exception:
            pass
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
        updated: dict[str, bool] = {}
        # Snapshot previous rqids for synchronization check
        prev_rqids: dict[str, int | None] = {pid: self._last_rqids.get(pid) for pid in self.agent_ids}
        for pid in self.agent_ids:
            opp = "player_1" if pid == "player_0" else "player_0"
            # Try to retrieve the latest battle update for this player
            try:
                self._logger.debug(
                    "[ACTWAIT] %s will race_get qsize=%d wait=%s try_again=%s finished_evt=%s",
                    pid,
                    self._battle_queues[pid].qsize(),
                    self._env_players[pid]._waiting.is_set(),
                    self._env_players[opp]._trying_again.is_set(),
                    self._finished_events[pid].is_set(),
                )
            except Exception:
                pass
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

            updated[pid] = battle is not None and (
                battle.last_request is not self._last_requests.get(pid)
            )
            # Snapshot per-pid summary after retrieval path is decided
            try:
                lr = getattr(battle, "last_request", None)
                rq = lr.get("rqid") if isinstance(lr, dict) else None
                self._logger.debug(
                    "[STEP] src pid=%s finished=%s turn=%s obj=%s tag=%s rqid=%s updated=%s",
                    pid,
                    getattr(battle, "finished", None),
                    getattr(battle, "turn", None),
                    hex(id(battle)) if battle else None,
                    getattr(battle, "battle_tag", None),
                    rq,
                    updated[pid],
                )
            except Exception:
                pass
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

        def _needs_update(b: Any, prev: int | None) -> bool:
            # finished battles do not require updates
            if getattr(b, "finished", False):
                return False
            lr = getattr(b, "last_request", None)
            if isinstance(lr, dict):
                # Consider teampreview as not-ready
                if lr.get("teamPreview"):
                    return True
                # If we have a previous rqid, require it to change
                if prev is not None:
                    curr = lr.get("rqid")
                    return curr == prev
                return False
            # If we don't have a structured request yet, wait
            return True

        pending: set[str] = set(
            pid for pid in self.agent_ids if _needs_update(battles[pid], prev_rqids.get(pid))
        )

        # RQID 同期: 以降は同一 Battle オブジェクトの in-place 更新を待つ。
        # 追加のバトル更新キューのドレインは行わず、PSClient による
        # battle.last_request の更新が反映されるまで短いスリープで再評価する。
        if pending:
            deadline = time.monotonic() + float(self.timeout)
            while pending and time.monotonic() < deadline:
                for pid in list(pending):
                    b = battles[pid]
                    if not _needs_update(b, prev_rqids.get(pid)):
                        pending.discard(pid)
                    else:
                        # Trace unchanged state for diagnostics
                        try:
                            lr = getattr(b, "last_request", None)
                            rq = lr.get("rqid") if isinstance(lr, dict) else None
                            tp = bool(lr.get("teamPreview")) if isinstance(lr, dict) else False
                            wt = bool(lr.get("wait")) if isinstance(lr, dict) else False
                            fs = bool(lr.get("forceSwitch")) if isinstance(lr, dict) else False
                            self._logger.debug(
                                "[RQSYNC] %s pending obj=%s tag=%s type=%s rqid=%s prev=%s",
                                pid,
                                hex(id(b)) if b else None,
                                getattr(b, "battle_tag", "?"),
                                ("teampreview" if tp else ("wait" if wt else ("force" if fs else "normal"))),
                                rq,
                                prev_rqids.get(pid),
                            )
                        except Exception:
                            pass
                # Allow PSClient/IPC pumps to process and update in-place
                if pending:
                    asyncio.run_coroutine_threadsafe(asyncio.sleep(0.05), POKE_LOOP).result()

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
                # Prefer player's own close_connection if available (handles IPC/WS)
                try:
                    if hasattr(p, "close_connection"):
                        asyncio.run_coroutine_threadsafe(
                            p.close_connection(), POKE_LOOP
                        ).result()
                        continue
                except Exception:
                    pass
                # Fallback to PSClient.stop_listening only when websocket exists
                try:
                    ps_client = getattr(p, "ps_client", None)
                    stop_listening = getattr(ps_client, "stop_listening", None)
                    websocket_obj = getattr(ps_client, "websocket", None)
                    if callable(stop_listening) and websocket_obj is not None:
                        asyncio.run_coroutine_threadsafe(
                            stop_listening(), POKE_LOOP
                        ).result()
                except Exception:
                    pass
            self._env_players.clear()

        for q in self._action_queues.values():
            asyncio.run_coroutine_threadsafe(q.join(), POKE_LOOP).result()
        for q in self._battle_queues.values():
            asyncio.run_coroutine_threadsafe(q.join(), POKE_LOOP).result()
        self._action_queues.clear()
        self._battle_queues.clear()

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
    
