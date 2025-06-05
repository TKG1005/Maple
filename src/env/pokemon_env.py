"""Gymnasium environment skeleton for Pokémon battles."""

from __future__ import annotations

from typing import Any, Tuple


import numpy as np

import gymnasium as gym
import asyncio
import threading
import time


class PokemonEnv(gym.Env):
    """A placeholder Gymnasium environment for Pokémon battles."""

    # Step12: エピソード終了判定に利用する最大ターン数
    MAX_TURNS: int = 100

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

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
        """Reset the environment and start a new battle."""

        super().reset(seed=seed)

        # poke_env は開発環境によってはインストールされていない場合があるため、
        # メソッド内で遅延インポートする。
        try:
            from poke_env.player import Player
            from poke_env.ps_client.server_configuration import ServerConfiguration
            from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

        except Exception as exc:  # pragma: no cover - ランタイム用
            raise RuntimeError(
                "poke_env package is required to run PokemonEnv"
            ) from exc

        # 対戦用のプレイヤーは初回のみ生成し、2 回目以降はリセットする。
        if not hasattr(self, "_env_player"):

            from pathlib import Path

            self_env = self

            class EnvPlayer(Player):
                """Simple player used internally by the environment."""

                async def choose_move(self, battle):
                    """Waits for an action index from the queue and returns an order."""
                    action_idx = await self_env._action_queue.get()
                    return self_env.action_helper.action_index_to_order(
                        self, battle, action_idx
                    )

            team_path = Path(__file__).resolve().parents[2] / "config" / "my_team.txt"
            try:
                team = team_path.read_text()
            except OSError:  # pragma: no cover - デバッグ用
                team = None

            self._env_player = EnvPlayer(
                battle_format="gen9ou",
                server_configuration=LocalhostServerConfiguration,
                team=team,
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

        # バトルオブジェクトが生成されるまで待機
        while not self._env_player.battles:
            time.sleep(0.1)

        # 開始したばかりのバトルオブジェクトを取得
        battle = next(iter(self._env_player.battles.values()))
        self._current_battle = battle
        observation = self.state_observer.observe(battle)

        info: dict = {"battle_tag": battle.battle_tag}
        return observation, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """Take a step in the environment using the given action."""
        # まずアクションをキューに送信
        self._action_queue.put_nowait(int(action))

        # 現在のバトル状態を取得
        battle = getattr(self, "_current_battle", None)

        if battle is not None:
            observation = self.state_observer.observe(battle)
            reward = self._calc_reward(battle)
            terminated, truncated = self._check_episode_end(battle)
            info: dict = {"turn": getattr(battle, "turn", 0)}
        else:  # バトルが取得できない場合はダミー値
            observation = self.observation_space.sample()
            reward = 0.0
            terminated = False
            truncated = False
            info = {}

        return observation, reward, terminated, truncated, info

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

    # Step12: エピソード終了判定ユーティリティ
    def _check_episode_end(self, battle: Any) -> Tuple[bool, bool]:
        """Return termination flags based on battle state and turn count."""

        terminated = getattr(battle, "finished", False)
        truncated = False
        if not terminated and getattr(battle, "turn", 0) > self.MAX_TURNS:
            truncated = True
        return terminated, truncated

    def render(self) -> None:
        """Render the environment if applicable."""
        return None

    def close(self) -> None:
        """Clean up resources used by the environment."""
        pass
