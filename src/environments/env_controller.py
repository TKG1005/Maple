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
from typing import Optional, TYPE_CHECKING, Dict, Any, Tuple
import types

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
        
        self._loop.set_exception_handler(
    lambda loop, ctx: logger.error("UNHANDLED ASYNCIO EXCEPTION: %s", ctx["message"])
)
        


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

    # --------------------------------------------------------------
    # 追加: 既存 WebSocket / listen タスクを確実に閉じるヘルパ
    # --------------------------------------------------------------
    async def _cleanup_ws(self) -> None:
        # ――― ① listen() タスク & WebSocket を完全停止 ―――
        async def _stop(client):
            coro = getattr(client, "_listening_coroutine", None)
            if coro and not coro.done():
                try:
                    await client.stop_listening()
                except Exception:
                    pass
                if not coro.done():
                    coro.cancel()
                    try:
                        await coro
                    except asyncio.CancelledError:
                        pass

        await asyncio.gather(
            _stop(self._player.ps_client),
            _stop(self._opponent.ps_client),
            return_exceptions=True,
        )

        # ――― ② Showdown! 側が同じユーザ名を解放するのを待つ ―――
        await asyncio.sleep(0.2)   # 0.05 秒では短いケースがあったため延長

        # ――― ③ ログインフラグをクリア（再接続時に必須） ―――
        for client in (self._player.ps_client, self._opponent.ps_client):
            client.logged_in.clear()
        
    # ------------------------------------------------------------------
    # 非同期実装本体 -----------------------------------------------------
    # ------------------------------------------------------------------

    async def _reset_async(
        self, seed: Optional[int], options: Optional[Dict[str, Any]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """poke-env にチャレンジを送り，最初の観測を返す。"""

        await self._cleanup_ws()
        if seed is not None:
            np.random.seed(seed)

        await self._ensure_listeners()
        await asyncio.gather(
            self._player.ps_client.wait_for_login(),
            self._opponent.ps_client.wait_for_login(),
        )

        self._player.reset_battles()
        self._opponent.reset_battles()

        await self._launch_challenge()
        await self._wait_until(lambda: self._current_battle is not None, RESET_TIMEOUT)
        await self._wait_until(self._battle_ready, 2.0)

        obs = self._state_observer.observe(self._current_battle)
        self._last_observation = obs
        return obs, {
            "battle_tag": self._current_battle.battle_tag,
            "turn": self._current_battle.turn,
            "opponent": self._opponent.username,
        }

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
        await self._cleanup_ws()
        logger.info("Backend closed.")
    # ------------------------------------------------------------------
    # helper utilities --------------------------------------------------
    # ------------------------------------------------------------------

    async def _ensure_listeners(self):
        """EnvPlayer / opponent の listen() を開始する．"""
        for client in (self._player.ps_client, self._opponent.ps_client):
            coro = getattr(client, "_listening_coroutine", None)
            if coro and not coro.done():
                if client.logged_in.is_set():
                    # 既に稼働中ならスキップ
                    continue
                # 古い listen() が終わっていない場合は一度イベントループを回す
                await asyncio.sleep(0)
            client._listening_coroutine = self._loop.create_task(client.listen())

    async def _launch_challenge(self):
        """1 戦だけチャレンジを送信・受諾．"""

        self._loop.create_task(
            self._player.send_challenges(self._opponent.username, 1)
        )
        self._loop.create_task(
            self._opponent.accept_challenges(None, 1)
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
        # [FIX] PM 形式チャレンジが改行付きで届くと poke-env 本家の
        # _handle_message が無視してしまうバグへのワークアラウンド。
        # 改行有無に関係なく `/challenge` を検出し _handle_challenge_request
        # を呼ぶラッパを差し込み、元実装の機能はそのまま保持する。
        # ------------------------------------------------------------------
        original_handle_message = self.ps_client._handle_message  # type: ignore[attr-defined]

        async def _patched_handle_message(msg: str, *,
                                          _orig=original_handle_message,
                                          _player=self):
            # PM のみ監視し、行単位で再分割して challenge を検出
            if msg.startswith("|pm|"):
                for tokens in (m.split("|") for m in msg.split("\n") if m):
                    if len(tokens) > 5 and tokens[4].startswith("/challenge"):
                        try:
                            # Player メソッドを利用してキューへ追加
                            await _player._handle_challenge_request(tokens)  # type: ignore[arg-type]
                        except Exception:
                            # 失敗しても他の処理に影響させない
                            pass
            # 既存ロジックへ委譲
            await _orig(msg)  # type: ignore[arg-type]

        # PSClient インスタンスへモンキーパッチ
        self.ps_client._handle_message = _patched_handle_message  # type: ignore


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
        trigger_tags = {"faint","request", "win", "lose", "tie", "error"}
        if any(len(m) > 1 and m[1] in trigger_tags for m in split_messages):
            self._battle_update_event.set()

    async def wait_for_battle_update(self, battle: Battle):
        logger.debug(f"Entering wait_for_battle_update: turn={battle.turn}, rqid={battle.last_request.get('rqid', -1)},_wait={battle._wait},forceSwitch={battle.force_switch}")
        prev_turn = battle.turn
        prev_rqid = battle.last_request.get("rqid", -1)
        while True:
            await self._battle_update_event.wait()
            
            self._battle_update_event.clear()
            if battle.finished:
                break
            #強制交代リクエストを優先して処理
            if battle.force_switch: 
                if battle.move_on_next_request:
                    logger.info("[wait] 強制交代リクエストを検出")
                break
            if (
                battle.turn > prev_turn or
                battle.last_request.get("rqid", -1) > prev_rqid
            ) and not battle._wait:
                break