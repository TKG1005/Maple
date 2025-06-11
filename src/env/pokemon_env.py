"""Gymnasium environment skeleton for Pokémon battles."""

from __future__ import annotations

from typing import Any, Tuple


import numpy as np

import gymnasium as gym
import asyncio
import threading
import time
import logging

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
        self._action_queue: asyncio.Queue[int] = asyncio.Queue()
        # EnvPlayer から受け取る battle オブジェクト用キュー
        self._battle_queue: asyncio.Queue[Any] = asyncio.Queue()

        self._agent = None  # MapleAgent を後から登録するための保持先

        self.opponent_player = opponent_player
        self.state_observer = state_observer
        self.action_helper = action_helper
        self.rng = np.random.default_rng(seed)

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
        self._action_queue = asyncio.Queue()
        self._battle_queue = asyncio.Queue()

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
                battle_format="gen9ou",
                server_configuration=LocalhostServerConfiguration,
                team=team,
                log_level=logging.DEBUG,
            )
        else:
            # 既存プレイヤーのバトル履歴をクリア
            self._env_player.reset_battles()

        if hasattr(self.opponent_player, "reset_battles"):
            self.opponent_player.reset_battles()

        # 対戦を非同期で開始 (ローカルの Showdown サーバー使用)
        async def start_battle() -> None:
            await asyncio.gather(
                self._env_player.send_challenges(
                    self.opponent_player.username,
                    n_challenges=1,
                    to_wait=self.opponent_player.ps_client.logged_in,
                ),
                self.opponent_player.accept_challenges(
                    self._env_player.username, n_challenges=1
                ),
            )

        self._battle_thread = threading.Thread(
            target=lambda: asyncio.run(start_battle()),
            daemon=True,
        )
        self._battle_thread.start()

        # バトル生成を待った後、チーム選択リクエストを待機
        while not self._env_player.battles:
            time.sleep(0.1)

        while self._battle_queue.empty():
            time.sleep(0.1)

        battle = self._battle_queue.get_nowait()
        observation = self.state_observer.observe(battle)

        info: dict = {
            "battle_tag": battle.battle_tag,
            "request_teampreview": True,
        }
        return observation, info

    def step(self, action: Any) -> Tuple[Any, dict, float, bool, dict]:
        """Send ``action`` to :class:`EnvPlayer` and wait for the next state."""

        # アクションインデックスをキューへ投入
        self._action_queue.put_nowait(int(action))

        # EnvPlayer から次の battle オブジェクトが届くまで待機
        while self._battle_queue.empty():
            time.sleep(0.1)

        battle = self._battle_queue.get_nowait()

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
        pass
