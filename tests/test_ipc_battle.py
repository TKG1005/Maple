import logging
import asyncio
import pytest
from src.sim.ipc_battle import IPCBattle


class DummyComm:
    """Dummy communicator for testing IPCBattle."""
    def __init__(self):
        self.sent = []
        self.connected = False

    async def is_alive(self):  # pragma: no cover
        return self.connected

    async def connect(self):  # pragma: no cover
        self.connected = True

    async def send_message(self, message):  # pragma: no cover
        self.sent.append(message)

    async def receive_message(self):  # pragma: no cover
        # Simulate fallback battle_state response
        return {"battle_state": {"foo": "bar"}}

    async def get_battle_state(self, battle_id):  # pragma: no cover
        # Simulate direct get_battle_state response
        return {"baz": "qux"}


def test_send_and_get_battle_state(caplog):
    """Test that IPCBattle sends commands and retrieves state correctly."""
    caplog.set_level(logging.DEBUG)
    logger = logging.getLogger("test_ipc_battle")
    comm = DummyComm()
    # Initialize IPCBattle
    battle = IPCBattle(
        battle_id="test123",
        username="user",
        logger=logger,
        communicator=comm
    )
    async def runner():
        # send command: should auto-connect and record message
        await battle.send_battle_command("move 1")
        assert comm.connected, "Communicator should be connected after send"
        assert comm.sent, "Message list should not be empty"
        sent_msg = comm.sent[-1]
        assert sent_msg.get("type") == "battle_command"
        assert sent_msg.get("battle_id") == "test123"
        assert sent_msg.get("command") == "move 1"
        # get battle state: should use communicator.get_battle_state()
        state = await battle.get_battle_state()
        assert state == {"baz": "qux"}, f"Unexpected state: {state}"
    # Run the async test
    asyncio.run(runner())