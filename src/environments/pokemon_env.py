# -*- coding: utf-8 -*-
"""
リファクタリング版 pokemon_env.py
===============================

主な改善点
-----------
1. **関心事の分離**
   * Gym API を担当する `PokemonEnv` と，poke‑env との非同期通信を担当する
     内部クラス `_AsyncPokemonBackend` に責務を分割しました。
2. **デバッグ出力の整理**
   * すべての `print()` を `logging` モジュールへ置き換え。モジュール直下に
     `logger = logging.getLogger(__name__)` を定義し，ライブラリ側では値を
     変更しません（利用者側で `logging.basicConfig()` 等を設定）。
3. **ハードコード定数の集約**
   * 報酬やタイムアウトなどを『設定値（定数）』として冒頭にまとめました。
4. **poke‑env 依存の局所化**
   * poke‑env を直接触るコードを `_AsyncPokemonBackend` と `EnvPlayer`
     に閉じ込め，`PokemonEnv` は Gym の API と入出力に集中しました。

※ PEP8 に準拠して命名を変更した箇所があります。
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import threading
import numpy as np
from gymnasium import spaces
from websockets.exceptions import ConnectionClosedOK

# --- poke‑env ---------------------------------------------------------------
from poke_env.environment.battle import Battle
from poke_env.exceptions import ShowdownException
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import BattleOrder
from poke_env.ps_client.account_configuration import AccountConfiguration

# --- maple project ----------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.state.state_observer import StateObserver
from .env_controller import _AsyncPokemonBackend 

# 型ヒントだけ必要な場合は import guard を付けると circular import を防げる
from src.action import action_helper as action_helper_module  # noqa: E402  # type: ignore

# ---------------------------------------------------------------------------
# 定数（設定値）
# ---------------------------------------------------------------------------
ACTION_SPACE_SIZE: int = 10
# 報酬
REWARD_WIN: float = 1.0
REWARD_LOSS: float = -1.0
REWARD_TIE: float = 0.0
REWARD_INVALID: float = -0.01
# タイムアウト
RESET_TIMEOUT: float = 30.0  # [sec]
STEP_TIMEOUT: float = 10.0  # [sec]

logger = logging.getLogger(__name__)

__all__ = ["PokemonEnv"]


class PokemonEnv(gym.Env):
    """Gymnasium 互換ラッパー．

    * 同期 API (`reset` / `step` / `close`) だけを公開し，実作業は
      `_AsyncPokemonBackend` に委譲します。
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        opponent_player: Player,
        state_observer: StateObserver,
        action_helper,
        *,
        battle_format: str = "gen9ou",
        team_pascal: Optional[str] = None,
        player_username: str = "MapleEnvPlayer",
    ) -> None:
        super().__init__()

        # --- Gym 空間定義 ---
        obs_dim = state_observer.get_observation_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # 10 = 技4 + テラスタル4 + 交代2
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # --- バックエンド ---
        self._backend = _AsyncPokemonBackend(
            env_ref=self,
            opponent_player=opponent_player,
            state_observer=state_observer,
            action_helper=action_helper,
            battle_format=battle_format,
            team_pascal=team_pascal,
            player_username=player_username,
        )
        
        #close() 呼び出し済みかどうかを管理
        self._closed = False
        
    @property
    def current_battle(self):
        """
        現在進行中の Battle オブジェクト (まだ開始していなければ None) を返す。
        test_env_step_loop.py や学習ループから直接参照される。
        """
        return self._backend._current_battle

    # ---------------------------------------------------------------------
    # Gym API (同期)
    # ---------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] | None = None, options: Optional[Dict[str, Any]] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:  # type: ignore[override]
        """環境リセット (同期)."""
        return self._backend.sync_reset(seed, options)

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:  # type: ignore[override]
        """
        1 ステップ実行 (同期)。Backend で STEP_TIMEOUT を超えた場合は
        truncated=True を付与して返す。
        """
        obs, rew, term, trunc, info = self._backend.sync_step(action)
        # Backend から "timeout" フラグが上がってきたら truncated 扱い
        if info.get("error") == "timeout":
            trunc = True
        return obs, rew, term, trunc, info

    def render(self, mode: str = "human") -> None:  # noqa: D401
        if mode != "human":
            raise NotImplementedError
        self._backend.render()

    def close(self):  # noqa: D401  # type: ignore[override]
        if not self._closed:
            self._backend.sync_close()
            self._closed = True

    # ------------------------ 追加ここから ------------------------
    def __del__(self):
        """
        ガーベジコレクタが環境を破棄する際に未 close なら自動で close。
        例外が出ても interpreter 終了を妨げないよう握り潰す。
        """
        try:
            # CPython の GC スレッドと衝突しないようロック
            lock = threading.Lock()
            with lock:
                self.close()
        except Exception:
            pass
    # ------------------------ 追加ここまで ------------------------

