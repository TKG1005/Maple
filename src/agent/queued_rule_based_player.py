import asyncio

from src.agents.rule_based_player import RuleBasedPlayer
from src.action.action_helper import action_index_to_order


class QueuedRuleBasedPlayer(RuleBasedPlayer):
    """Rule-based player with optional async action queue override."""

    def __init__(self, *args, action_queue: asyncio.Queue[int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_queue: asyncio.Queue[int] = action_queue or asyncio.Queue()

    async def choose_move(self, battle):
        """Return a BattleOrder from queue or fallback to rule-based logic."""
        try:
            idx = self.action_queue.get_nowait()
        except asyncio.QueueEmpty:
            return super().choose_move(battle)
        return action_index_to_order(self, battle, idx)
