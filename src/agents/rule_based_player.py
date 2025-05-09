# rule_based_player.py

import numpy as np
from poke_env.player import Player
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
#修正↓: 正しいPokemonTypeクラスをインポート
from poke_env.environment.pokemon_type import PokemonType

# 以前のタスクで作成したモジュールをインポート
from src.state.state_observer import StateObserver
from src.action.action_helper import get_available_actions, action_index_to_order

class RuleBasedPlayer(Player):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            # 実行時のカレントディレクトリがプロジェクトルートであることを想定
            # rule_based_player.py が src/agents/ にある場合、
            # パスは "../../config/state_spec.yml" のようになる可能性があります。
            # ご自身のプロジェクト構造に合わせてパスを修正してください。
            self.observer = StateObserver("config/state_spec.yml") # ★パス確認・修正★
            print("StateObserver initialized successfully.")
        except FileNotFoundError:
            print("Error: state_spec.yml not found. Please check the path.")
            raise
    def choose_move(self, battle: Battle):
        """
        ルールに基づいて行動を選択し、そのプロセスをログに出力します。
        ルール1: 相手に効果抜群の利用可能な技があれば、その中で最も威力の高い技を選択。
        フォールバック: 上記に該当がなければ、利用可能な行動の中からランダムに選択。
        """
        player_username = self.username
        current_turn = battle.turn
        my_active_pokemon = battle.active_pokemon
        opponent_active_pokemon = battle.opponent_active_pokemon

        # --- ログヘッダー ---
        print(f"\n--- Turn {current_turn} ({player_username}) ---")
        if my_active_pokemon:
            print(f"My Active: {my_active_pokemon.species} (HP: {my_active_pokemon.current_hp_fraction * 100:.1f}%)")
        if opponent_active_pokemon:
            print(f"Opponent Active: {opponent_active_pokemon.species} (HP: {opponent_active_pokemon.current_hp_fraction * 100:.1f}%)")
        # --- 状態観測 ---
        try:
            current_state_vector = self.observer.observe(battle) # 必要であればコメントアウト解除
            # print(f"State observed. Vector length: {len(current_state_vector)}") # 詳細すぎる場合はコメントアウト
            pass # StateObserver自体は初期化時に確認しているので、ここでは省略可
        except Exception as e:
            print(f"LOG: Error observing state: {e}")
            print("LOG: Falling back to random move due to state observation error.")
            return self.choose_random_move(battle)

        # --- 行動取得 ---
        try:
            available_action_mask, available_action_mapping = get_available_actions(battle)
            # print(f"LOG: Available action mask: {available_action_mask}") # 詳細すぎる場合はコメントアウト
            print(f"LOG: Available action mapping: {available_action_mapping}")
            if not available_action_mapping: # マッピングが空なら選択肢なし
                print("LOG: No actions available in mapping. Choosing a random move (fallback).")
                return self.choose_random_move(battle)
        except Exception as e:
            print(f"LOG: Error getting available actions: {e}")
            print("LOG: Falling back to random move due to action getting error.")
            return self.choose_random_move(battle)

        chosen_action_index = -1
        applied_rule = "N/A"

        # --- ルールベースの意思決定 ---
        if opponent_active_pokemon and opponent_active_pokemon.type_1: # 相手ポケモンとそのタイプが判明している場合
            best_super_effective_move_index = -1
            max_power = -1
            best_move_obj = None

            for action_idx, (action_type, original_move_idx) in available_action_mapping.items():
                if action_type == "move": # 通常技のみ考慮 (テラスタルは別途ルール化が必要なら追加)
                    if not available_action_mask[action_idx]: # マスクで不可ならスキップ
                        continue
                    
                    move: Move = battle.available_moves[original_move_idx]
                    if not move: continue

                    type_multiplier = opponent_active_pokemon.damage_multiplier(move)

                    if type_multiplier >= 2: # 効果抜群
                        current_move_power = move.base_power
                        if current_move_power > max_power:
                            max_power = current_move_power
                            best_super_effective_move_index = action_idx
                            best_move_obj = move
            
            if best_super_effective_move_index != -1 and best_move_obj:
                chosen_action_index = best_super_effective_move_index
                applied_rule = f"Super Effective (Power: {max_power}, Move: {best_move_obj.id}, TypeMultiplier: {opponent_active_pokemon.damage_multiplier(best_move_obj):.1f}x vs {opponent_active_pokemon.species})"
            else:
                applied_rule = "No Super Effective Move Found"
        elif opponent_active_pokemon and not opponent_active_pokemon.type_1:
            applied_rule = "Opponent Type Unknown"
        else: # opponent_active_pokemon is None
            applied_rule = "Opponent Pokemon Info Missing"


        # --- フォールバック (ランダム選択) ---
        if chosen_action_index == -1:
            possible_action_indices = [
                idx for idx in available_action_mapping.keys()
                if available_action_mask[idx] # マスクが1のものを選択
            ]
            
            if not possible_action_indices: # 利用可能な行動が本当にない場合
                print(f"LOG: ({applied_rule}) -> No valid actions available based on mask and mapping. Choosing random move (ultimate fallback).")
                return self.choose_random_move(battle)

            chosen_action_index = np.random.choice(possible_action_indices)
            applied_rule += f" -> Fallback to Random (from {len(possible_action_indices)} options)"
            # どの行動がランダムで選ばれたかを追加でログ
            randomly_selected_action_details = available_action_mapping.get(chosen_action_index)
            if randomly_selected_action_details:
                action_type, original_idx = randomly_selected_action_details
                if action_type == "move":
                    move_name = battle.available_moves[original_idx].id
                    applied_rule += f": Chose move {move_name}"
                elif action_type == "terastal":
                    move_name = battle.available_moves[original_idx].id
                    applied_rule += f": Chose terastal move {move_name}"
                elif action_type == "switch":
                    poke_name = battle.available_switches[original_idx].species
                    applied_rule += f": Chose switch to {poke_name}"


        # --- 最終決定とログ ---
        try:
            final_order = action_index_to_order(self, battle, chosen_action_index)
            action_type_log, original_idx_log = available_action_mapping[chosen_action_index]
            action_details_log = ""
            if action_type_log == "move":
                action_details_log = f"Move: {battle.available_moves[original_idx_log].id}"
            elif action_type_log == "terastal":
                 action_details_log = f"Terastal Move: {battle.available_moves[original_idx_log].id}"
            elif action_type_log == "switch":
                action_details_log = f"Switch: {battle.available_switches[original_idx_log].species}"

            print(f"LOG: Rule Applied: '{applied_rule}'")
            print(f"LOG: Chosen Action Index: {chosen_action_index} ({action_details_log})")
            print(f"LOG: Order to send: '{final_order}'")
            print(f"--- End of Turn {current_turn} ({player_username}) Decisions ---")
            return final_order
        except KeyError: # chosen_action_index がマッピングにない場合（基本的には起こらないはず）
            print(f"LOG: Error - Chosen action index {chosen_action_index} not in available_action_mapping. This should not happen.")
            print(f"LOG: available_action_mapping was: {available_action_mapping}")
            print(f"LOG: available_action_mask was: {available_action_mask}")
            print("LOG: Falling back to random move due to critical mapping error.")
            return self.choose_random_move(battle)
        except ValueError as e: # action_index_to_order でのエラー
             print(f"LOG: Error converting action index {chosen_action_index} to order: {e}")
             print("LOG: Falling back to random move due to order conversion error.")
             return self.choose_random_move(battle)
        except Exception as e:
            print(f"LOG: An unexpected error occurred during final order decision: {e}")
            print("LOG: Falling back to random move.")
            return self.choose_random_move(battle)

    def teampreview(self, battle: Battle):
        """
        チームプレビュー時の選択。ログはシンプルに。
        """
        player_username = self.username
        team_size = len(battle.team) if battle.team else 0
        num_to_select = min(3, team_size) # シングル6→3ルール想定

        print(f"\n--- Teampreview ({player_username}) ---")
        if num_to_select > 0:
            # 最初の num_to_select 体を選択 (ここでは単純な戦略)
            selected_pokemon_indices = [str(i + 1) for i in range(num_to_select)]
            order = "/team " + "".join(selected_pokemon_indices)
            print(f"LOG: Selected team order: {order} (first {num_to_select} Pokemon)")
            return order
        else:
            # チームが空などのエッジケース
            print("LOG: Warning: Team is empty or invalid during teampreview. Defaulting to /team 123")
            return "/team 123"

# --- クラス定義はここまで ---

if __name__ == '__main__':
    print("RuleBasedPlayer class defined with super effective move logic (import fixed).")
    # 対戦実行スクリプトでテストしてください。