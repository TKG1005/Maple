from __future__ import annotations

import logging
from typing import Dict, Optional

from .ipc_battle_controller import IPCBattleController


class _ControllerRegistry:
    def __init__(self) -> None:
        # Full migration: manage controllers by room_tag only
        self._by_room: Dict[str, IPCBattleController] = {}
        # Protect concurrent access from multiple threads/tasks
        import threading
        self._lock = threading.Lock()

    def get(self, room_tag: str) -> Optional[IPCBattleController]:
        return self._by_room.get(room_tag)
    def get_or_create(self, node_script_path: str, room_tag: str, logger: Optional[logging.Logger] = None) -> IPCBattleController:
        """Get or create a controller keyed by `room_tag`.

        Note: Full migration uses `room_tag` as the canonical identifier for
        controllers; callers should pass the room_tag here.
        """
        with self._lock:
            ctrl = self._by_room.get(room_tag)
            if ctrl is not None:
                return ctrl

            ctrl = IPCBattleController(node_script_path=node_script_path, battle_id=room_tag, logger=logger)
            self._by_room[room_tag] = ctrl
            return ctrl

    def get_or_create_shared(self, node_script_path: str, pool_key: str = "__ipc_pool__", logger: Optional[logging.Logger] = None) -> IPCBattleController:
        """Get or create a shared (reusable) controller.

        This returns a controller instance that can be reused sequentially across
        multiple battles. The controller's battle_id will be updated at
        `create_battle` time. Concurrency per process remains 1 in Phase 1.
        """
        with self._lock:
            ctrl = self._by_room.get(pool_key)
            if ctrl is not None:
                return ctrl
            ctrl = IPCBattleController(node_script_path=node_script_path, battle_id=pool_key, logger=logger)
            self._by_room[pool_key] = ctrl
            return ctrl

    def remove(self, room_tag: str) -> None:
        with self._lock:
            self._by_room.pop(room_tag, None)


ControllerRegistry = _ControllerRegistry()
