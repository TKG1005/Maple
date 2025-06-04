import asyncio
import json
import pytest

pytest.importorskip("poke_env")

from run_battle import main

@pytest.mark.asyncio
async def test_run_battle_result():
    result = await main()
    assert result["turns"] > 0
    assert result["winner"] in {"p1", "p2"}
