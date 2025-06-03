from pathlib import Path

from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from src.agents.my_simple_player import MySimplePlayer


def main() -> None:
    """Run a single battle between MySimplePlayer and RandomPlayer."""
    team_path = Path(__file__).resolve().parents[1] / "config" / "my_team.txt"
    try:
        team = team_path.read_text()
    except OSError:
        team = None

    player = MySimplePlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        team=team,
    )
    opponent = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    player.play_against(opponent, n_battles=1)


if __name__ == "__main__":
    main()
