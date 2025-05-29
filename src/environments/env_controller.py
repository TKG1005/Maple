# -*- coding: utf-8 -*-
"""
非同期バックエンドと EnvPlayer

* _AsyncPokemonBackend : poke-env と通信し Gym API へ同期ラッパを提供
* EnvPlayer            : poke-env.Player を RL 用に拡張
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Optional, TYPE_CHECKING, Dict, Any, Tuple

import numpy as np
from poke_env.environment.battle import Battle
from poke_env.exceptions import ShowdownException
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import BattleOrder
from poke_env.ps_client.account_configuration import AccountConfiguration
from websockets.exceptions import ConnectionClosedOK

import gymnasium as gym

# --- maple project ----------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.state.state_observer import StateObserver
from src.action import action_helper as action_helper_module

if TYPE_CHECKING:  # 循環 import を避ける
    from .pokemon_env import PokemonEnv

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
ACTION_SPACE_SIZE: int = 10
REWARD_WIN, REWARD_LOSS, REWARD_TIE, REWARD_INVALID = 1.0, -1.0, 0.0, -0.01
RESET_TIMEOUT, STEP_TIMEOUT = 30.0, 10.0  # [sec]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,           # ← DEBUG に下げる
    stream=sys.stdout,
)

__all__ = ["_AsyncPokemonBackend", "EnvPlayer"]


class _AsyncPokemonBackend:
    """poke-env と非同期通信し、PokemonEnv から呼ばれる同期 API を提供する。"""

    # ------------------------------------------------------------------
    # コンストラクタ
    # ------------------------------------------------------------------

    def __init__(
        self,
        env_ref: "PokemonEnv",
        opponent_player: Player,
        state_observer: StateObserver,
        action_helper=action_helper_module,
        *,
        battle_format: str,
        team_pascal: Optional[str],
        player_username: str,
    ) -> None:
        
        self._env = env_ref
        self._state_observer = state_observer
        self._action_helper = action_helper
        self._current_battle: Optional[Battle] = None
        self._battle_is_over: bool = True
        self._last_observation: Optional[np.ndarray] = None

        # --- asyncio イベントループ確保 -------------------------------
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        # --- EnvPlayer / opponent --------------------------------------
        env_account = AccountConfiguration(player_username, None)
        self._player = EnvPlayer(
            account_configuration=env_account,
            battle_format=battle_format,
            team=team_pascal,
            log_level=logging.DEBUG,
            env_ref=self,
        )
        self._opponent = opponent_player

    # ------------------------------------------------------------------
    # 同期ラッパー
    # ------------------------------------------------------------------

    def sync_reset(
        self, seed: Optional[int], options: Optional[Dict[str, Any]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """同期 reset."""
        if self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(
                self._reset_async(seed, options), self._loop
            )
            return fut.result()
        return self._loop.run_until_complete(self._reset_async(seed, options))

    def sync_step(
        self, action_idx: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """同期 step."""
        if self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(
                self._step_async(action_idx), self._loop
            )
            return fut.result()
        return self._loop.run_until_complete(self._step_async(action_idx))

    def sync_close(self):  # noqa: D401
        """同期 close."""
        if self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._close_async(), self._loop)
            fut.result()
        else:
            self._loop.run_until_complete(self._close_async())

    # ------------------------------------------------------------------
    # public helper -----------------------------------------------------
    # ------------------------------------------------------------------

    def render(self) -> None:
        """テキストベースの簡易レンダラ．"""
        battle = self._current_battle
        if not battle:
            logger.info("Battle not started.")
            return
        if self._battle_is_over:
            logger.info("Battle is over. Call reset() to start a new one.")
            return

        logger.info("\n--- Turn %d ---", battle.turn)
        my_active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon
        if my_active:
            mv_str = ", ".join(
                f"{m.id}({m.current_pp}/{m.max_pp})" for m in my_active.moves.values()
            )
            logger.info(
                "My %s HP: %.1f%% | Moves: %s",
                my_active.species,
                my_active.current_hp_fraction * 100,
                mv_str,
            )
        if opp_active:
            logger.info(
                "Opp %s HP: %.1f%%",
                opp_active.species,
                opp_active.current_hp_fraction * 100,
            )

    # ------------------------------------------------------------------
    # 非同期実装本体 -----------------------------------------------------
    # ------------------------------------------------------------------

    async def _reset_async(
        self, seed: Optional[int], options: Optional[Dict[str, Any]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """poke‑env にチャレンジを送り，最初の観測を返す．"""
        if seed is not None:
            np.random.seed(seed)

        # --- listen() を起動 -----------------------------------------
        await self._ensure_listeners()

        # --- ログイン完了待ち ----------------------------------------
        await asyncio.gather(
            self._player.ps_client.wait_for_login(),
            self._opponent.ps_client.wait_for_login(),
        )

        # --- 前回のバトル情報を初期化 -------------------------------
        self._player.reset_battles()
        self._opponent.reset_battles()
        self._current_battle = None
        self._battle_is_over = False

        # --- チャレンジ送信 ------------------------------------------
        await self._launch_challenge()

        # --- Battle オブジェクトが用意されるのを待つ -----------------
        await self._wait_until(lambda: self._current_battle is not None, RESET_TIMEOUT)

        # --- request が出揃うまで再度待機 ---------------------------
        await self._wait_until(self._battle_ready, 2.0)

        obs = self._state_observer.observe(self._current_battle)  # type: ignore[arg-type]
        self._last_observation = obs
        info = {
            "battle_tag": self._current_battle.battle_tag,  # type: ignore[union-attr]
            "turn": self._current_battle.turn,  # type: ignore[union-attr]
            "opponent": self._opponent.username,
        }
        return obs, info

    async def _step_async(
        self, action_idx: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """1 ターン分の行動を送り，サーバ応答を待って結果を返す．"""
        if not self._current_battle or self._battle_is_over:
            logger.error("Battle not ready or already finished.")
            dummy = (
                self._last_observation
                if self._last_observation is not None
                else np.zeros(self._env.observation_space.shape, dtype=np.float32)
            )
            return dummy, 0.0, True, False, {"error": "battle_not_started"}

        # --- action → BattleOrder -----------------------------------
        try:
            order: BattleOrder = self._action_helper.action_index_to_order(
                self._player, self._current_battle, action_idx
            )
        except ValueError as e:
            logger.warning("Invalid action %d: %s", action_idx, e)
            obs = self._state_observer.observe(self._current_battle)
            self._last_observation = obs
            return obs, REWARD_INVALID, False, False, {"error": str(e)}

        self._player.set_next_action_for_battle(self._current_battle, order)

        # --- ターン終了まで待機 -------------------------------------
        try:
            await asyncio.wait_for(
                self._player.wait_for_battle_update(self._current_battle),
                timeout=STEP_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for battle update.")
        except ShowdownException as e:
            logger.error("ShowdownException: %s", e)
            self._battle_is_over = True
            obs = self._state_observer.observe(self._current_battle)
            self._last_observation = obs
            return obs, REWARD_LOSS, True, True, {"error": "showdown"}

        # --- 観測・報酬計算 -----------------------------------------
        obs = self._state_observer.observe(self._current_battle)
        self._last_observation = obs
        terminated = self._current_battle.finished
        reward = 0.0
        if terminated:
            self._battle_is_over = True
            if self._current_battle.won:
                reward = REWARD_WIN
            elif getattr(self._current_battle, "tied", False):
                reward = REWARD_TIE
            else:
                reward = REWARD_LOSS
        info = {
            "turn": self._current_battle.turn,
            "won": self._current_battle.won if terminated else None,
        }
        return obs, reward, terminated, False, info

    async def _close_async(self):  # noqa: D401
        """WebSocket を閉じる．"""
        logger.info("Closing PokemonEnv backend …")
        tasks = []
        if self._player.ps_client.is_listening:
            tasks.append(self._player.ps_client.stop_listening())
        if self._opponent.ps_client.is_listening:
            tasks.append(self._opponent.ps_client.stop_listening())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Backend closed.")

    # ------------------------------------------------------------------
    # helper utilities --------------------------------------------------
    # ------------------------------------------------------------------

    async def _ensure_listeners(self):
        """EnvPlayer / opponent の listen() を開始する．"""
        if not self._player.ps_client.is_listening:
            self._loop.create_task(self._player.ps_client.listen())
        if not self._opponent.ps_client.is_listening:
            self._loop.create_task(self._opponent.ps_client.listen())

    async def _launch_challenge(self):
        """1 戦だけチャレンジを送信・受諾．"""
        self._loop.create_task(
            self._player.send_challenges(self._opponent.username, 1)
        )
        self._loop.create_task(
            self._opponent.accept_challenges(self._player.username, 1)
        )

    async def _wait_until(self, cond, timeout: float):
        """`cond()` が True になるか timeout するまでポーリング．"""
        start = self._loop.time()
        while not cond():
            if self._loop.time() - start > timeout:
                raise TimeoutError("Condition wait timed out")
            await asyncio.sleep(0.1)

    def _battle_ready(self) -> bool:
        b = self._current_battle
        if not b:
            return False
        return b.active_pokemon is not None and b.available_moves



# ---------------------------------------------------------------------------
# EnvPlayer
# ---------------------------------------------------------------------------


class EnvPlayer(Player):
    """poke‑env の Player を RL 用に拡張したクラス (ほぼ元実装)."""

    def __init__(
        self,
        account_configuration: AccountConfiguration,
        battle_format: str,
        team: Optional[str] = None,
        log_level: int | None = None,
        *,
        env_ref: _AsyncPokemonBackend | None = None,
    ) -> None:
        super().__init__(
            account_configuration=account_configuration,
            battle_format=battle_format,
            team=team,
            log_level=log_level,
            start_listening=False,
        )
        self._backend_ref = env_ref
        self._next_action_future: Optional[asyncio.Future] = None
        self._last_rqid = -1 
        self._battle_update_event = asyncio.Event()
        # poke‑env コールバック差し替え
        self.ps_client._handle_battle_message = self._handle_battle_message  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # RL インタフェース -------------------------------------------------
    # ------------------------------------------------------------------

    def choose_move(self, battle: Battle):
        current_rqid = battle.last_request.get("rqid", -1)
        logger.debug(f"[choose_move] rqid={current_rqid}, last={self._last_rqid}")
        if current_rqid < self._last_rqid:
            logger.warning(f"[choose_move] 古いrequest rqid={current_rqid} を無視 (last={self._last_rqid})")
            return asyncio.get_event_loop().create_future() 
        
        if self._backend_ref and self._backend_ref._current_battle is None:
            self._backend_ref._current_battle = battle
        # ログ: Future既存チェック
        logger.debug(f"[choose_move] Turn {battle.turn}, _wait={battle._wait}, "
                     f"Future exists={bool(self._next_action_future)} (done={self._next_action_future.done() if self._next_action_future else None})")
        
        if current_rqid == self._last_rqid:
            if self._next_action_future and not self._next_action_future.done():
                return self._next_action_future
        
        if self._next_action_future and not self._next_action_future.done():
            logger.debug(f"[choose_move] Reusing existing pending Future.")
            return self._next_action_future
        self._last_rqid = current_rqid
        loop = asyncio.get_event_loop()
        self._next_action_future = loop.create_future()
        logger.debug(f"[choose_move] Created new Future {id(self._next_action_future)} for turn {battle.turn}.")
        return self._next_action_future

    def set_next_action_for_battle(self, battle: Battle, order: BattleOrder):
        logger.debug(f"[set_next_action] Called at turn {battle.turn}, _wait={battle._wait}, move_on_next_request={battle.move_on_next_request}. "
                     f"Future exists={bool(self._next_action_future)} (done={self._next_action_future.done() if self._next_action_future else None})")
        if self._next_action_future and not self._next_action_future.done():
            logger.debug(f"[set_next_action] Setting result on Future {id(self._next_action_future)}.")
            self._next_action_future.set_result(order)
        else:
            logger.warning("Future not pending; using backup path.")
            self._next_action_future = asyncio.get_event_loop().create_future()
            logger.debug(f"[set_next_action] Created new backup Future {id(self._next_action_future)} and set result immediately.")
            self._next_action_future.set_result(order)

    # ------------------------------------------------------------------
    # メッセージハンドラ ------------------------------------------------
    # ------------------------------------------------------------------

    async def _handle_battle_message(self, split_messages):  # type: ignore[override]
        await super()._handle_battle_message(split_messages)
        trigger_tags = {"request", "win", "lose", "tie", "error"}
        if any(len(m) > 1 and m[1] in trigger_tags for m in split_messages):
            self._battle_update_event.set()

    async def wait_for_battle_update(self, battle: Battle):
        logger.debug(f"Entering wait_for_battle_update: turn={battle.turn}, rqid={battle.last_request.get('rqid', -1)},_wait={battle._wait}")
        prev_turn = battle.turn
        prev_rqid = battle.last_request.get("rqid", -1)
        while True:
            await self._battle_update_event.wait()
            self._battle_update_event.clear()
            if battle.finished:
                break
            if (
                battle.turn > prev_turn or
                battle.last_request.get("rqid", -1) > prev_rqid
            ) and not battle._wait:
                if battle.force_switch:
                    logger.info("[wait] 強制交代リクエストを検出")
                    if battle.available_switches:
                        switch_in = battle.available_switches[0]
                        self.set_next_action_for_battle(battle, BattleOrder(switch_in))
                        # switch 完了までループ継続
                        prev_turn = battle.turn
                        prev_rqid = battle.last_request.get("rqid", -1)
                        continue
                    else:
                        logger.warning("[wait] 交代先がいない")
                        break
                break