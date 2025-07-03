import numpy as np
from poke_env.data import GenData
from poke_env.environment.battle import AbstractBattle
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.move_category import MoveCategory


class TypeMatchupFeatureExtractor:
    def __init__(self, gen: int = 9):
        self.type_chart = GenData.from_gen(gen).type_chart
    
    def _log2_damage_multiplier(self, multiplier: float) -> float:
        """Convert damage multiplier (0-4x) to log2 scale (-10 to 2)."""
        if multiplier == 0.0:
            return -10.0  # Use large negative value instead of -inf to avoid NaN
        return np.log2(multiplier)

    def extract(self, battle: AbstractBattle) -> np.ndarray:
        my_active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon
        opp_bench = [p for p in battle.opponent_team.values() if p != opp_active and p.active is False]

        # 自分の技と相手ポケモンの相性
        move_matchups = []
        my_moves = sorted(my_active.moves.values(), key=lambda m: m.id)[:4]
        for move in my_moves:
            # 対相手アクティブ
            move_matchups.append(self._log2_damage_multiplier(self._get_damage_multiplier(move, opp_active)))
            # 対相手ベンチ
            for i in range(2):
                if i < len(opp_bench):
                    move_matchups.append(self._log2_damage_multiplier(self._get_damage_multiplier(move, opp_bench[i])))
                else:
                    move_matchups.append(0.0)  # log2(1.0) = 0.0

        # 相手のタイプ一致技と自分ポケモンの相性
        stab_matchups = []
        my_bench = [p for p in battle.team.values() if p != my_active and p.active is False]

        if opp_active:
            opp_types = [opp_active.type_1, opp_active.type_2]
            if opp_active.is_terastallized:
                opp_types = [opp_active.tera_type, None]

            for opp_type in opp_types:
                if opp_type:
                    # 対自分アクティブ
                    stab_matchups.append(self._log2_damage_multiplier(opp_type.damage_multiplier(my_active.type_1, my_active.type_2, type_chart=self.type_chart)))
                    # 対自分ベンチ
                    for i in range(2):
                        if i < len(my_bench):
                            stab_matchups.append(self._log2_damage_multiplier(opp_type.damage_multiplier(my_bench[i].type_1, my_bench[i].type_2, type_chart=self.type_chart)))
                        else:
                            stab_matchups.append(0.0)  # log2(1.0) = 0.0
                else:
                    stab_matchups.extend([0.0] * 3)  # log2(1.0) = 0.0
        else:
            stab_matchups.extend([0.0] * 6)  # log2(1.0) = 0.0

        # 12 + 6 = 18要素のベクトル
        return np.array(move_matchups + stab_matchups, dtype=np.float32)

    def _get_damage_multiplier(self, move: Pokemon, pokemon: Pokemon) -> float:
        if not pokemon or not move.type:
            return 1.0
        
        if move.category == MoveCategory.STATUS:
            return 1.0

        if pokemon.is_terastallized:
            return move.type.damage_multiplier(pokemon.tera_type, None, type_chart=self.type_chart)
        
        return move.type.damage_multiplier(pokemon.type_1, pokemon.type_2, type_chart=self.type_chart)
