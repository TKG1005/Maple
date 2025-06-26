"""Gymnasium environment skeleton for Pokémon battles."""

from __future__ import annotations

from typing import Any, Tuple
import warnings


import numpy as np

import gymnasium as gym
from gymnasium.spaces import Dict
import asyncio
import logging
from pathlib import Path
import yaml
from poke_env.concurrency import POKE_LOOP

from .env_player import EnvPlayer


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

        # マルチエージェント用のエージェントID
        self.agent_ids = ("player_0", "player_1")

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

        # チームプレビューで得た手持ちポケモン一覧と選択されたポケモン種別
        self._team_rosters: dict[str, list[str]] = {
            agent_id: [] for agent_id in self.agent_ids
        }
        self._selected_species: dict[str, set[str]] = {
            agent_id: set() for agent_id in self.agent_ids
        }

        # Battle.last_request の更新を追跡するためのキャッシュ
        self._last_requests: dict[str, Any] = {
            agent_id: None for agent_id in self.agent_ids
        }
        # 直近に計算した行動マスク
        self._latest_masks: tuple[np.ndarray, ...] | None = None

    # ------------------------------------------------------------------
    # Agent interaction utilities
    # ------------------------------------------------------------------
    def register_agent(self, agent: Any, player_id: str = "player_0") -> None:
        """Register the controlling :class:`MapleAgent` for a given player."""
        if not hasattr(self, "_agents"):
            self._agents: dict[str, Any] = {}
        self._agents[player_id] = agent

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
        The mask is filtered using ``_selected_species`` so that switches to
        unselected Pokémon become unavailable.
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
            selected = self._selected_species.get(player_id)
            if selected:
                for idx, detail in mapping.items():
                    if (
                        detail.get("type") == "switch"
                        and detail.get("id") not in selected
                    ):
                        mask[idx] = 0
            self._action_mappings[player_id] = mapping
            return mask, mapping

        # without details use internal computation routine
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
        self._need_action = {agent_id: True for agent_id in self.agent_ids}
        self._action_mappings = {agent_id: {} for agent_id in self.agent_ids}
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

        # 対戦用のプレイヤーは初回のみ生成し、2 回目以降はリセットする。
        if not hasattr(self, "_env_players"):
            from pathlib import Path

            team_path = (
                Path(__file__).resolve().parents[2] / "config" / "my_team.txt"
            )
            try:
                team = team_path.read_text()
            except OSError:  # pragma: no cover - デバッグ用
                team = None

            self._env_players = {
                "player_0": EnvPlayer(
                    self,
                    "player_0",
                    battle_format="gen9bssregi",
                    server_configuration=LocalhostServerConfiguration,
                    team=team,
                    log_level=logging.DEBUG,
                    save_replays=self.save_replays,
                )
            }
            if self.opponent_player is None:
                self._env_players["player_1"] = EnvPlayer(
                    self,
                    "player_1",
                    battle_format="gen9bssregi",
                    server_configuration=LocalhostServerConfiguration,
                    team=team,
                    log_level=logging.DEBUG,
                    save_replays=self.save_replays,
                )
            else:
                self._env_players["player_1"] = self.opponent_player
        else:
            # 既存プレイヤーのバトル履歴をクリア
            for p in self._env_players.values():
                if hasattr(p, "reset_battles"):
                    p.reset_battles()

        if self.opponent_player is not None and hasattr(
            self.opponent_player, "reset_battles"
        ):
            self.opponent_player.reset_battles()

        # 対戦開始処理を poke-env のイベントループで同期実行
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
        self._selected_species["player_0"] = set(self._team_rosters["player_0"])
        self._team_rosters["player_1"] = [p.species for p in battle1.team.values()]
        self._selected_species["player_1"] = set(self._team_rosters["player_1"])

        self._current_battles = {"player_0": battle0, "player_1": battle1}
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
        self._latest_masks = masks

        if hasattr(self, "single_agent_mode"):
            if return_masks:
                return observation[self.agent_ids[0]], info, masks[0]
            return observation[self.agent_ids[0]], info

        if return_masks:
            return observation, info, masks
        return observation, info

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

    def _race_get(
        self,
        queue: asyncio.Queue[Any],
        *events: asyncio.Event,
    ) -> Any | None:
        """Return queue item or ``None`` if any event fires first."""

        async def _race() -> Any | None:
            get_task = asyncio.create_task(queue.get())
            wait_tasks = [asyncio.create_task(e.wait()) for e in events]
            done, pending = await asyncio.wait(
                {get_task, *wait_tasks},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for p in pending:
                p.cancel()
            if get_task in done:
                return get_task.result()
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
            [0 if disabled else 1 for _, (_, _, disabled) in sorted(action_mapping.items())],
            dtype=np.int8,
        )

    def _compute_all_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current legal action masks for both players."""

        masks: list[np.ndarray] = []
        for pid in self.agent_ids:
            battle = self.get_current_battle(pid)
            if battle is None:
                masks.append(np.zeros(self.ACTION_SIZE, dtype=np.int8))
                self._action_mappings[pid] = {}
                continue

            mapping = self.action_helper.get_action_mapping(battle)
            mask = self._build_action_mask(mapping)

            self._logger.debug(
                "[DBG] %s: %d moves, %d switches (force_switch=%s)",
                pid,
                len(getattr(battle, "available_moves", [])),
                len(getattr(battle, "available_switches", [])),
                getattr(battle, "force_switch", False),
            )

            switches_info = [
                f"{getattr(p, 'species', '?')}"
                f"(HP:{getattr(p, 'current_hp_fraction', 0) * 100:.1f}%"
                f", fainted={getattr(p, 'fainted', False)}"
                f", active={getattr(p, 'active', False)})"
                for p in getattr(battle, "available_switches", [])
            ]
            self._logger.debug("[DBG] %s available_switches: %s", pid, switches_info)

            selected = self._selected_species.get(pid)
            if selected:
                for idx, (atype, sub_idx, disabled) in mapping.items():
                    if atype == "switch" and not disabled:
                        try:
                            pkmn = battle.available_switches[sub_idx]
                        except IndexError:
                            continue
                        if pkmn.species not in selected:
                            mask[idx] = 0

            self._action_mappings[pid] = mapping
            masks.append(mask)

        return tuple(masks)  # type: ignore[return-value]

    def step(self, action_dict: dict[str, int | str], *, return_masks: bool = True):
        """マルチエージェント形式で1ステップ進める。

        ``return_masks`` を ``True`` にすると、戻り値の末尾に各プレイヤーの
        行動マスクを含むタプルを追加で返す。
        """

        for agent_id in self.agent_ids:
            if agent_id not in action_dict:
                raise ValueError(f"{agent_id} action required")
            if not self._need_action.get(agent_id, True):
                continue
            act = action_dict[agent_id]

            if isinstance(act, str):
                asyncio.run_coroutine_threadsafe(
                    self._action_queues[agent_id].put(act), POKE_LOOP
                ).result()
                if act.startswith("/team"):
                    import re

                    indices = [int(x) - 1 for x in re.findall(r"\d", act)]
                    roster = self._team_rosters.get(agent_id, [])
                    self._selected_species[agent_id] = {
                        roster[i] for i in indices if 0 <= i < len(roster)
                    }
            else:
                battle = self._current_battles.get(agent_id)
                if battle is None:
                    raise ValueError(f"No current battle for {agent_id}")
                mapping = self.action_helper.get_action_mapping(battle)
                self._action_mappings[agent_id] = mapping

                switch_info = [
                    f"({getattr(p, 'species', '?')},"
                    f"{getattr(p, 'current_hp_fraction', 0) * 100:.1f}%,"
                    f"{getattr(p, 'fainted', False)},"
                    f"{getattr(p, 'active', False)})"
                    for p in getattr(battle, "available_switches", [])
                ]
                self._logger.debug(
                    "[DBG] %s mapping=%s sw=%d force=%s info=%s",
                    agent_id,
                    mapping,
                    len(getattr(battle, "available_switches", [])),
                    getattr(battle, "force_switch", False),
                    switch_info,
                )

                DisabledErr = getattr(
                    self.action_helper, "DisabledMoveError", ValueError
                )
                try:
                    order = self.action_helper.action_index_to_order_from_mapping(
                        self._env_players[agent_id],
                        battle,
                        int(act),
                        mapping,
                    )
                except DisabledErr:
                    mask = self._build_action_mask(mapping)
                    valid = [i for i, m in enumerate(mask) if m == 1]
                    if not valid:
                        raise
                    new_idx = int(self.rng.choice(valid))
                    self._logger.warning(
                        "%s selected disabled action %s; fallback to %s",
                        agent_id,
                        act,
                        new_idx,
                    )
                    order = self.action_helper.action_index_to_order_from_mapping(
                        self._env_players[agent_id],
                        battle,
                        new_idx,
                        mapping,
                    )
                asyncio.run_coroutine_threadsafe(
                    self._action_queues[agent_id].put(order), POKE_LOOP
                ).result()

        # 次の状態を待機する前に対戦タスクの状態を確認
        if hasattr(self, "_battle_task") and self._battle_task.done():
            exc = self._battle_task.exception()
            if exc is not None:
                raise exc

        battles: dict[str, Any] = {}
        updated: dict[str, bool] = {}
        for pid in self.agent_ids:
            opp = "player_1" if pid == "player_0" else "player_0"
            battle = self._race_get(
                self._battle_queues[pid],
                self._env_players[pid]._waiting,
                self._env_players[opp]._trying_again,
            )
            self._env_players[pid]._waiting.clear()
            if battle is None:
                self._env_players[opp]._trying_again.clear()
                battle = self._current_battles[pid]
                self._need_action[pid] = False
            else:
                self._current_battles[pid] = battle
                self._need_action[pid] = True
            battles[pid] = battle
            updated[pid] = battle is not None and (
                battle.last_request is not self._last_requests.get(pid)
            )

        if all(updated.get(pid, False) for pid in self.agent_ids):
            masks = self._compute_all_masks()
            for pid in self.agent_ids:
                self._last_requests[pid] = self._current_battles[pid].last_request
            self._latest_masks = masks
        else:
            if self._latest_masks is None:
                masks = self._compute_all_masks()
                self._latest_masks = masks
            else:
                masks = self._latest_masks

        observations = {
            pid: self.state_observer.observe(battles[pid]) for pid in self.agent_ids
        }
        rewards = {pid: self._calc_reward(battles[pid]) for pid in self.agent_ids}
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
    def _calc_reward(self, battle: Any) -> float:
        """Return +1 if the battle is won, -1 if lost, otherwise 0."""

        # battle.finished はバトルが終了したかどうかを示す poke-env の属性
        if not getattr(battle, "finished", False):
            return 0.0

        # battle.won が True なら勝利、False なら敗北とみなす
        if getattr(battle, "won", False):
            return 1.0
        return -1.0

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
                asyncio.run_coroutine_threadsafe(
                    p.ps_client.stop_listening(), POKE_LOOP
                ).result()
            self._env_players.clear()

        for q in self._action_queues.values():
            asyncio.run_coroutine_threadsafe(q.join(), POKE_LOOP).result()
        for q in self._battle_queues.values():
            asyncio.run_coroutine_threadsafe(q.join(), POKE_LOOP).result()
        self._action_queues.clear()
        self._battle_queues.clear()
