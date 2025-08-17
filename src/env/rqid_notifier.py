from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    version: int
    last_rqid: Optional[int]
    cond: asyncio.Condition
    closed: bool = False


class RqidNotifier:
    """Versioned notifier for rQID updates (single-loop usage).

    Notes
    -----
    - 想定ループ: POKE_LOOP 内での利用（クロススレッドは未対応の骨組み）。
    - トリガ: 「rqid が更新＝発火」。rqid 変化がなければ通知しない。
    - 失敗ポリシー: 本コンポーネントは待機のタイムアウトで例外を投げる（フォールバックなし）。
    """

    def __init__(self) -> None:
        self._entries: Dict[str, _Entry] = {}

    def register_battle(self, player_id: str, initial_rqid: Optional[int] = None) -> None:
        if player_id not in self._entries:
            self._entries[player_id] = _Entry(version=0, last_rqid=initial_rqid, cond=asyncio.Condition())

    def close_battle(self, player_id: str) -> None:
        ent = self._entries.get(player_id)
        if ent is None:
            return
        ent.closed = True
        # wake all waiters
        try:
            async def _notify_all():
                async with ent.cond:
                    ent.cond.notify_all()
            # Best-effort: schedule on current loop
            asyncio.get_event_loop().create_task(_notify_all())
        except Exception:
            pass

    async def wait_for_rqid_change(self, player_id: str, baseline_rqid: Optional[int], timeout: float) -> int:
        """Wait until last_rqid != baseline_rqid or timeout.

        Returns the new rqid when changed. Raises TimeoutError on timeout.
        """
        if player_id not in self._entries:
            # Lazy register with unknown baseline
            self.register_battle(player_id, initial_rqid=None)
        ent = self._entries[player_id]

        start = time.monotonic()
        # Fast path
        if ent.last_rqid is not None and ent.last_rqid != baseline_rqid:
            try:
                dt_ms = int((time.monotonic() - start) * 1000)
                logger.info(
                    "[METRIC] tag=RQWAIT_FAST pid=%s baseline=%s new_rqid=%s elapsed_ms=%d",
                    player_id,
                    baseline_rqid,
                    ent.last_rqid,
                    dt_ms,
                )
            except Exception:
                pass
            return int(ent.last_rqid)

        deadline = start + float(timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    dt_ms = int((time.monotonic() - start) * 1000)
                    logger.info(
                        "[METRIC] tag=RQWAIT_TIMEOUT pid=%s baseline=%s elapsed_ms=%d",
                        player_id,
                        baseline_rqid,
                        dt_ms,
                    )
                except Exception:
                    pass
                raise TimeoutError(f"rqid wait timed out for {player_id} baseline={baseline_rqid}")
            async with ent.cond:
                try:
                    await asyncio.wait_for(ent.cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    continue
            # Woken up; check condition
            if ent.closed:
                raise RuntimeError(f"notifier closed for {player_id}")
            if ent.last_rqid is not None and ent.last_rqid != baseline_rqid:
                try:
                    dt_ms = int((time.monotonic() - start) * 1000)
                    logger.info(
                        "[METRIC] tag=RQWAIT_OK pid=%s baseline=%s new_rqid=%s elapsed_ms=%d",
                        player_id,
                        baseline_rqid,
                        ent.last_rqid,
                        dt_ms,
                    )
                except Exception:
                    pass
                return int(ent.last_rqid)

    async def publish_rqid_update(self, player_id: str, rqid: Optional[int], meta: Optional[dict] = None) -> None:
        if player_id not in self._entries:
            self.register_battle(player_id, initial_rqid=rqid)
        ent = self._entries[player_id]
        # Only notify on actual change
        if rqid is None or rqid == ent.last_rqid:
            return
        ent.last_rqid = rqid
        ent.version += 1
        try:
            logger.info(
                "[RQPUB] pid=%s rqid_new=%s version=%d",
                player_id,
                rqid,
                ent.version,
            )
        except Exception:
            pass
        async with ent.cond:
            ent.cond.notify_all()


_GLOBAL: Optional[RqidNotifier] = None


def get_global_rqid_notifier() -> RqidNotifier:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = RqidNotifier()
    return _GLOBAL

