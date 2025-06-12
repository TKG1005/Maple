"""Gymnasium environment skeleton for Pokémon battles."""

from __future__ import annotations

from typing import Any, Tuple


import numpy as np

import gymnasium as gym
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
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(state_dim,),
            dtype=np.float32,
        )
        # Action indices are represented as a discrete space.
        self.action_space = gym.spaces.Discrete(self.ACTION_SIZE)

    # ------------------------------------------------------------------
    # Agent interaction utilities
    # ------------------------------------------------------------------
    def register_agent(self, agent: Any) -> None:
        """Register the controlling :class:`MapleAgent`."""
        self._agent = agent

    def process_battle(self, battle: Any) -> int:
        """Create an observation and available action mapping for ``battle``.

        The resulting state vector and action mapping are sent to the registered
        :class:`MapleAgent` which returns an action index.
        """
        if self._agent is None:
            raise RuntimeError("Agent not registered")

        observation = self.state_observer.observe(battle)

        # battle 情報から利用可能な行動マッピングを生成
        _, action_mapping = self.action_helper.get_available_actions_with_details(
            battle
        )

        action_idx = self._agent.select_action(observation, action_mapping)
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
            self._env_player.battle_against(self.opponent_player, n_battles=1),
            POKE_LOOP,
        )

        # チーム選択リクエストを待機
        battle = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(self._battle_queue.get(), self.timeout),
            POKE_LOOP,
        ).result()
        POKE_LOOP.call_soon_threadsafe(self._battle_queue.task_done)
        observation = self.state_observer.observe(battle)

        info: dict = {
            "battle_tag": battle.battle_tag,
            "request_teampreview": True,
        }
        return observation, info

    def step(self, action: Any) -> Tuple[Any, dict, float, bool, dict]:
        """Send ``action`` to :class:`EnvPlayer` and wait for the next state."""

        # アクション (行動インデックスまたはチーム選択文字列) をキューへ投入
        if isinstance(action, str):
            asyncio.run_coroutine_threadsafe(
                self._action_queue.put(action), POKE_LOOP
            ).result()
        else:
            asyncio.run_coroutine_threadsafe(
                self._action_queue.put(int(action)), POKE_LOOP
            ).result()

        # EnvPlayer から次の battle オブジェクトが届くまで待機
        battle = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(self._battle_queue.get(), self.timeout),
            POKE_LOOP,
        ).result()
        POKE_LOOP.call_soon_threadsafe(self._battle_queue.task_done)

        observation = self.state_observer.observe(battle)
        _, action_mapping = self.action_helper.get_available_actions_with_details(
            battle
        )
        reward = self._calc_reward(battle)
        done: bool = bool(getattr(battle, "finished", False))
        info: dict = {}

        return observation, action_mapping, reward, done, info

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
