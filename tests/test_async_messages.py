from typing import Any
import asyncio


class DummyOrder:
    def __init__(self, message: str) -> None:
        self.message = message


class DummyActionHelper:
    def action_index_to_order(self, player, battle, idx):
        return DummyOrder(f"/move {idx}")


class DummyEnv:
    def __init__(self, timeout: float = 1.0) -> None:
        self.timeout = timeout
        self._action_queues = {"player_0": asyncio.Queue()}
        self._battle_queues = {"player_0": asyncio.Queue()}
        self.action_helper = DummyActionHelper()


class DummyBattle:
    battle_tag = "test-battle"


class DummyEnvPlayer:
    def __init__(self, env: DummyEnv, player_id: str = "player_0") -> None:
        self._env = env
        self.player_id = player_id

    async def choose_move(self, battle: Any):
        await self._env._battle_queues[self.player_id].put(battle)
        action_data = await asyncio.wait_for(
            self._env._action_queues[self.player_id].get(), self._env.timeout
        )
        self._env._action_queues[self.player_id].task_done()
        if isinstance(action_data, int):
            return self._env.action_helper.action_index_to_order(
                self, battle, action_data
            )
        return action_data


def _run(coro):
    asyncio.run(asyncio.wait_for(coro, timeout=1.0))


def test_choose_move_with_delay():
    async def scenario():
        env = DummyEnv()
        player = DummyEnvPlayer(env, "player_0")
        battle = DummyBattle()

        async def provider():
            received = await env._battle_queues["player_0"].get()
            assert received is battle
            await asyncio.sleep(0.05)
            await env._action_queues["player_0"].put(1)

        provider_task = asyncio.create_task(provider())
        order = await player.choose_move(battle)
        assert order.message == "/move 1"
        await provider_task

    _run(scenario())


def test_choose_move_with_prequeued_action():
    async def scenario():
        env = DummyEnv()
        player = DummyEnvPlayer(env, "player_0")
        battle = DummyBattle()

        await env._action_queues["player_0"].put(2)

        async def provider():
            received = await env._battle_queues["player_0"].get()
            assert received is battle

        provider_task = asyncio.create_task(provider())
        order = await player.choose_move(battle)
        assert order.message == "/move 2"
        await provider_task

    _run(scenario())
