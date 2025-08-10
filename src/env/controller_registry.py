from __future__ import annotations

import logging
from typing import Dict, Optional

from .ipc_battle_controller import IPCBattleController


class _ControllerRegistry:
    def __init__(self) -> None:
        self._by_battle: Dict[str, IPCBattleController] = {}

    def get(self, battle_id: str) -> Optional[IPCBattleController]:
        return self._by_battle.get(battle_id)

    def get_or_create(self, node_script_path: str, battle_id: str, logger: Optional[logging.Logger] = None) -> IPCBattleController:
        ctrl = self._by_battle.get(battle_id)
        if ctrl is None:
            ctrl = IPCBattleController(node_script_path=node_script_path, battle_id=battle_id, logger=logger)
            self._by_battle[battle_id] = ctrl
        return ctrl

    def remove(self, battle_id: str) -> None:
        self._by_battle.pop(battle_id, None)


ControllerRegistry = _ControllerRegistry()

