"""Run a single battle and return the result as JSON serialisable data."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

# ``MySimplePlayer`` is our basic player defined in ``src.agents``. It selects
# actions randomly from the set of valid ones while printing some debug
# information. We use it for ``player_1`` in this script.
from src.agents.my_simple_player import MySimplePlayer


async def main() -> dict:
    # Player1 は ``MySimplePlayer`` を利用する。デバッグ用に手持ちチームの
    # 読み込みを試みるが、存在しなければ ``None`` を渡してランダムチームを
    # 使用する。
    team_file = Path(__file__).resolve().parents[1] / "config" / "my_team.txt"
    try:
        team_text = team_file.read_text()
    except OSError:
        team_text = None

    player_1 = MySimplePlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        team=team_text,
    )
    player_2 = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    await asyncio.gather(
        player_1.send_challenges(player_2.username, n_challenges=1),
        player_2.accept_challenges(player_1.username, n_challenges=1),
    )

    winner = "p1" if player_1.n_won_battles == 1 else "p2"
    battle = next(iter(player_1.battles.values()), None)
    turns = getattr(battle, "turn", 0)

    return {"winner": winner, "turns": turns}


if __name__ == "__main__":
    result = asyncio.run(main())
    print(json.dumps(result, ensure_ascii=False))
