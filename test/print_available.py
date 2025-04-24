# print_available.py

from poke_env.player import Player
from poke_env.environment.battle import Battle

class AvailableMovesChecker(Player):

    def choose_move(self, battle: Battle):
        print("\n=== 利用可能な技 ===")
        if battle.available_moves:
            for move in battle.available_moves:
                print(f"技名: {move.id}, 威力: {move.base_power}, 命中率: {move.accuracy}")
        else:
            print("利用可能な技はありません。")

        print("\n=== 交代可能なポケモン ===")
        if battle.available_switches:
            for poke in battle.available_switches:
                print(f"ポケモン名: {poke.species}, 残りHP割合: {poke.current_hp_fraction:.2f}")
        else:
            print("交代可能なポケモンはいません。")

        # 一旦ランダムに行動を返す（テストのため）
        return self.choose_random_move(battle)

if __name__ == "__main__":
    from poke_env import ShowdownServerConfiguration
    from poke_env.player.random_player import RandomPlayer

    checker = AvailableMovesChecker(battle_format="gen9randommbattle", log_level=25)
    opponent = RandomPlayer(battle_format="gen9randombattle", log_level=25)

    checker.battle_against(opponent, n_battles=1)
