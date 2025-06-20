"""Gymnasium environment skeleton for Pokémon battles."""

from __future__ import annotations

from typing import Any, Tuple


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
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.ACTION_SIZE = 10  # "gen9bss"ルールでは行動空間は10で固定
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
        self._action_mappings: dict[str, dict[int, tuple[str, int]]] = {
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
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[Any, dict]:
        """Reset the environment and start a new battle."""

        super().reset(seed=seed)

        # 前回エピソードのキューをクリア
        self._action_queues = {agent_id: asyncio.Queue() for agent_id in self.agent_ids}
        self._battle_queues = {agent_id: asyncio.Queue() for agent_id in self.agent_ids}
        self._need_action = {agent_id: True for agent_id in self.agent_ids}
        self._action_mappings = {agent_id: {} for agent_id in self.agent_ids}

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

            team_path = Path(__file__).resolve().parents[2] / "config" / "my_team.txt"
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

        if hasattr(self, "single_agent_mode"):
            return observation[self.agent_ids[0]], info

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
            return None

        result = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(_race(), self.timeout),
            POKE_LOOP,
        ).result()
        if result is not None:
            POKE_LOOP.call_soon_threadsafe(queue.task_done)
        return result

    def step(self, action_dict: dict[str, int | str]):
        """マルチエージェント形式で1ステップ進める。"""

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
                mapping = self._action_mappings.get(agent_id) or {}
                self._logger.debug("received action %s for %s", act, agent_id)
                self._logger.debug("current mapping for %s: %s", agent_id, mapping)
                if mapping:
                    order = self.action_helper.action_index_to_order_from_mapping(
                        self._env_players[agent_id],
                        self._current_battles[agent_id],
                        int(act),
                        mapping,
                    )
                else:
                    order = self.action_helper.action_index_to_order(
                        self._env_players[agent_id],
                        self._current_battles[agent_id],
                        int(act),
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
            mask, mapping = self.action_helper.get_available_actions(battle)
            self._logger.debug("available mask for %s: %s", pid, mask)
            self._logger.debug("available mapping for %s: %s", pid, mapping)
            selected = self._selected_species.get(pid)
            if selected:
                for idx, (atype, sub_idx) in mapping.items():
                    if atype == "switch":
                        try:
                            pkmn = battle.available_switches[sub_idx]
                        except IndexError:
                            continue
                        if pkmn.species not in selected:
                            mask[idx] = 0
            self._action_mappings[pid] = mapping

        observation = {
            pid: self.state_observer.observe(battles[pid]) for pid in self.agent_ids
        }
        rewards = {pid: self._calc_reward(battles[pid]) for pid in self.agent_ids}
        terminated = {}
        truncated = {}
        for pid in self.agent_ids:
            term, trunc = self._check_episode_end(battles[pid])
            terminated[pid] = term
            truncated[pid] = trunc
        if any(truncated.values()):
            rewards = {agent_id: 0.0 for agent_id in self.agent_ids}

        observations = observation
        term_flags = terminated
        trunc_flags = truncated
        infos = {agent_id: {} for agent_id in self.agent_ids}

        if hasattr(self, "single_agent_mode"):
            battle_sa = battles[self.agent_ids[0]]
            action_mask, mapping = self.action_helper.get_available_actions(battle_sa)
            selected = self._selected_species.get(self.agent_ids[0])
            if selected:
                for idx, (atype, sub_idx) in mapping.items():
                    if atype == "switch":
                        try:
                            pkmn = battle_sa.available_switches[sub_idx]
                        except IndexError:
                            continue
                        if pkmn.species not in selected:
                            action_mask[idx] = 0
            self._action_mappings[self.agent_ids[0]] = mapping
            done = terminated[self.agent_ids[0]] or truncated[self.agent_ids[0]]
            return (
                observation[self.agent_ids[0]],
                action_mask,
                rewards[self.agent_ids[0]],
                done,
                infos[self.agent_ids[0]],
            )

        return observations, rewards, term_flags, trunc_flags, infos

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
