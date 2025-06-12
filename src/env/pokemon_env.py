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
        opponent_player: Any,
        state_observer: Any,
        action_helper: Any,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.ACTION_SIZE = 10  # "gen9ou"ルールでは行動空間は10で固定
        self.MAX_TURNS = 100  # エピソードの最大ターン数

        # Step10: 非同期アクションキューを導入
        # 数値アクションだけでなく、チーム選択コマンドなどの文字列も
        # 取り扱えるよう ``Any`` 型のキューを使用する
        self._action_queue: asyncio.Queue[Any] = asyncio.Queue()
        # EnvPlayer から受け取る battle オブジェクト用キュー
        self._battle_queue: asyncio.Queue[Any] = asyncio.Queue()

        self._agent = None  # MapleAgent を後から登録するための保持先

        self.opponent_player = opponent_player
        self.state_observer = state_observer
        self.action_helper = action_helper
        self.rng = np.random.default_rng(seed)

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

    # ------------------------------------------------------------------
    # Agent interaction utilities
    # ------------------------------------------------------------------
    def register_agent(self, agent: Any) -> None:
        """Register the controlling :class:`MapleAgent`."""
        self._agent = agent

    def process_battle(self, battle: Any) -> int:
        """Create an observation and available action mask for ``battle``.

        The resulting state vector and action mask are sent to the registered
        :class:`MapleAgent` which returns an action index.
        """
        if self._agent is None:
            raise RuntimeError("Agent not registered")

        observation = self.state_observer.observe(battle)

        # battle 情報から利用可能な行動マスクを生成
        action_mask, _ = self.action_helper.get_available_actions_with_details(battle)

        action_idx = self._agent.select_action(observation, action_mask)
        return int(action_idx)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[Any, dict]:
        """Reset the environment and start a new battle."""

        super().reset(seed=seed)

        # 前回エピソードのキューをクリア
        self._action_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._battle_queue: asyncio.Queue[Any] = asyncio.Queue()

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
        if not hasattr(self, "_env_player"):
            from pathlib import Path

            team_path = Path(__file__).resolve().parents[2] / "config" / "my_team.txt"
            try:
                team = team_path.read_text()
            except OSError:  # pragma: no cover - デバッグ用
                team = None

            self._env_player = EnvPlayer(
                self,
                battle_format="gen9bssregi",
                server_configuration=LocalhostServerConfiguration,
                team=team,
                log_level=logging.WARNING,
            )
        else:
            # 既存プレイヤーのバトル履歴をクリア
            self._env_player.reset_battles()

        if hasattr(self.opponent_player, "reset_battles"):
            self.opponent_player.reset_battles()

        # 対戦開始処理を poke-env のイベントループで同期実行
        self._battle_task = asyncio.run_coroutine_threadsafe(
            asyncio.gather(
                self._env_player.play_against(self.opponent_player, n_battles=1),
                self.opponent_player.play_against(self._env_player, n_battles=1),
            ),
            POKE_LOOP,
        )

        # チーム選択リクエストを待機
        battle = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(self._battle_queue.get(), self.timeout),
            POKE_LOOP,
        ).result()
        POKE_LOOP.call_soon_threadsafe(self._battle_queue.task_done)
        self._current_battle = battle
        observation = self.state_observer.observe(battle)

        info: dict = {
            "battle_tag": battle.battle_tag,
            "request_teampreview": True,
        }

        if hasattr(self, "single_agent_mode"):
            return observation, info

        observations = {agent_id: observation for agent_id in self.agent_ids}
        return observations, info

    def step(self, action_dict: dict[str, int]):
        """マルチエージェント形式で1ステップ進める。"""

        # 入力アクションをキューへ登録
        player_action = action_dict.get("player_0")
        if player_action is None:
            raise ValueError("player_0 action required")

        if isinstance(player_action, str):
            asyncio.run_coroutine_threadsafe(
                self._action_queue.put(player_action), POKE_LOOP
            ).result()
        else:
            order = self.action_helper.action_index_to_order(
                self._env_player, self._current_battle, int(player_action)
            )
            asyncio.run_coroutine_threadsafe(
                self._action_queue.put(order), POKE_LOOP
            ).result()

        # 次の状態を待機
        battle = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(self._battle_queue.get(), self.timeout),
            POKE_LOOP,
        ).result()
        POKE_LOOP.call_soon_threadsafe(self._battle_queue.task_done)
        self._current_battle = battle

        observation = self.state_observer.observe(battle)
        reward_val = self._calc_reward(battle)
        terminated = bool(getattr(battle, "finished", False))
        truncated = getattr(battle, "turn", 0) > self.MAX_TURNS
        if truncated:
            reward_val = 0.0

        observations = {agent_id: observation for agent_id in self.agent_ids}
        rewards = {
            self.agent_ids[0]: reward_val,
            self.agent_ids[1]: -reward_val,
        }
        term_flags = {agent_id: terminated for agent_id in self.agent_ids}
        trunc_flags = {agent_id: truncated for agent_id in self.agent_ids}
        infos = {agent_id: {} for agent_id in self.agent_ids}

        if hasattr(self, "single_agent_mode"):
            action_mask, _ = self.action_helper.get_available_actions_with_details(
                battle
            )
            done = terminated or truncated
            return observation, action_mask, reward_val, done, infos[self.agent_ids[0]]

        return observations, rewards, term_flags, trunc_flags, infos

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
        """Render the environment if applicable."""
        return None

    def close(self) -> None:
        """Clean up resources used by the environment."""
        if hasattr(self, "_battle_task"):
            self._battle_task.cancel()
            asyncio.run_coroutine_threadsafe(self._battle_task, POKE_LOOP).result()
        if hasattr(self, "_env_player"):
            asyncio.run_coroutine_threadsafe(
                self._env_player.ps_client.stop_listening(), POKE_LOOP
            ).result()
        asyncio.run_coroutine_threadsafe(self._action_queue.join(), POKE_LOOP).result()
        asyncio.run_coroutine_threadsafe(self._battle_queue.join(), POKE_LOOP).result()
