"""Run a single battle between two :class:`RandomPlayer` instances."""

from __future__ import annotations

import asyncio

from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration


async def main() -> None:
    player_1 = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
    )
    player_2 = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    await asyncio.gather(
        player_1.send_challenges(player_2.username, n_challenges=1),
        player_2.accept_challenges(player_1.username, n_challenges=1),
    )

    result = "Win" if player_1.n_won_battles == 1 else "Loss"
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
