from __future__ import annotations

import logging
from typing import Dict, Optional, List

from .ipc_battle_controller import IPCBattleController


class _ControllerRegistry:
    def __init__(self) -> None:
        # Full migration: manage controllers by room_tag only
        self._by_room: Dict[str, IPCBattleController] = {}
        # Phase2: simple pool management per pool_key
        self._pools: Dict[str, List[IPCBattleController]] = {}
        # Track which controller a given battle_id is assigned to
        self._battle_assignment: Dict[str, IPCBattleController] = {}
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

    def get_shared_for_battle(
        self,
        node_script_path: str,
        pool_key: str,
        battle_id: str,
        max_processes: int = 2,
        logger: Optional[logging.Logger] = None,
    ) -> IPCBattleController:
        """Return a controller from a small pool with an upper bound.

        - If `battle_id` already assigned, return the same controller.
        - Otherwise, pick the first not-busy controller; if none and pool size
          is below `max_processes`, create a new controller.
        - As a last resort, return the controller with the lowest perceived load
          (for Phase1 sequential usage, this simply falls back to the first).
        """
        with self._lock:
            # Existing assignment for this battle?
            existing = self._battle_assignment.get(battle_id)
            if existing is not None:
                return existing

            pool = self._pools.get(pool_key)
            if pool is None:
                pool = []
                self._pools[pool_key] = pool

            # Prefer a not-busy controller
            for ctrl in pool:
                try:
                    if not getattr(ctrl, "_busy", False):
                        self._battle_assignment[battle_id] = ctrl
                        return ctrl
                except Exception:
                    continue

            # None available: create new if under limit
            if len(pool) < max_processes:
                ctrl = IPCBattleController(node_script_path=node_script_path, battle_id=pool_key, logger=logger)
                pool.append(ctrl)
                self._battle_assignment[battle_id] = ctrl
                return ctrl

            # Fallback: return first controller (may be busy; caller should queue)
            if pool:
                ctrl = pool[0]
                self._battle_assignment[battle_id] = ctrl
                return ctrl

            # As a defensive fallback create one
            ctrl = IPCBattleController(node_script_path=node_script_path, battle_id=pool_key, logger=logger)
            self._pools[pool_key] = [ctrl]
            self._battle_assignment[battle_id] = ctrl
            return ctrl

    def remove(self, room_tag: str) -> None:
        with self._lock:
            self._by_room.pop(room_tag, None)


ControllerRegistry = _ControllerRegistry()
