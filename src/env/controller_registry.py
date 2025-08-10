from __future__ import annotations

import logging
from typing import Dict, Optional

from .ipc_battle_controller import IPCBattleController


class _ControllerRegistry:
    def __init__(self) -> None:
        self._by_battle: Dict[str, IPCBattleController] = {}
        # Phase2: allow lookup by room_tag as well
        self._by_room: Dict[str, IPCBattleController] = {}

    def get(self, battle_id: str) -> Optional[IPCBattleController]:
        return self._by_battle.get(battle_id)

    def get_or_create(self, node_script_path: str, battle_id: str, room_tag: Optional[str] = None, logger: Optional[logging.Logger] = None) -> IPCBattleController:
        # Prefer battle_id lookup
        ctrl = self._by_battle.get(battle_id)
        if ctrl is not None:
            return ctrl

        # Fallback: if a room_tag is provided, try room-based lookup
        if room_tag:
            ctrl = self._by_room.get(room_tag)
            if ctrl is not None:
                return ctrl

        # Create new controller bound to battle_id; optionally register by room_tag
        ctrl = IPCBattleController(node_script_path=node_script_path, battle_id=battle_id, logger=logger)
        self._by_battle[battle_id] = ctrl
        if room_tag:
            self._by_room[room_tag] = ctrl
        return ctrl

    def remove(self, battle_id: str) -> None:
        ctrl = self._by_battle.pop(battle_id, None)
        # Also remove any room_tag mapping that points to this controller
        if ctrl is not None:
            room_tag = getattr(ctrl, "_room_tag", None)
            if isinstance(room_tag, str) and room_tag in self._by_room:
                self._by_room.pop(room_tag, None)


ControllerRegistry = _ControllerRegistry()
