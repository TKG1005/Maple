# src/action/action_helper.py
from typing import List, Tuple, Dict, Union, TypedDict # TypedDictを追加
import numpy as np
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.player.player import Player # Playerをインポート
from poke_env.environment.pokemon_type import PokemonType # 型ヒント用
# from src.action.action_masker import generate_action_mask # 既にインポート済みのはず

# (get_available_actions は変更なし、または詳細情報を返す版を追加)

# ActionDetail の型定義 (render用)
class ActionDetail(TypedDict):
    type: str
    name: str
    id: Union[str, int] # move.id や pokemon.species
    # 必要なら他の情報も追加 (例: move_type, move_power, pokemon_hp_fraction)

def get_available_actions_with_details(battle: Battle) -> Tuple[np.ndarray, Dict[int, ActionDetail]]:
    """
    Battle オブジェクトから現在可能な行動をマスクと詳細情報付きで取得します。
    renderメソッドでの表示やデバッグに利用できます。
    """
    mask, mapping = get_available_actions(battle) # 既存の関数を利用
    detailed_actions: Dict[int, ActionDetail] = {}

    available_moves_sorted = sorted(battle.available_moves, key=lambda m: m.id)
    available_switches_sorted = battle.available_switches # 元々リスト

    for action_index, (action_type, sub_index) in mapping.items():
        name = "Unknown"
        action_id = "unknown"
        if action_type == "move":
            if sub_index < len(available_moves_sorted):
                move = available_moves_sorted[sub_index]
                name = f"Move: {move.id} (PP: {move.current_pp}/{move.max_pp})"
                action_id = move.id
            else:
                name = "Invalid Move Index"
        elif action_type == "terastal":
            if sub_index < len(available_moves_sorted) and battle.can_tera:
                move = available_moves_sorted[sub_index]
                name = f"Terastal Move: {move.id} (PP: {move.current_pp}/{move.max_pp})"
                action_id = f"tera_{move.id}"
            else:
                name = "Invalid Terastal Move Index or Cannot Terastallize"
        elif action_type == "switch":
            if sub_index < len(available_switches_sorted):
                pkmn = available_switches_sorted[sub_index]
                name = f"Switch to: {pkmn.species} (HP: {pkmn.current_hp_fraction*100:.1f}%)"
                action_id = pkmn.species
            else:
                name = "Invalid Switch Index"

        detailed_actions[action_index] = {"type": action_type, "name": name, "id": action_id}
    return mask, detailed_actions


# action_index_to_order は Player インスタンスを引数に取るようにする
def action_index_to_order(player: Player, battle: Battle, action_index: int) -> str:
    """
    選択された action_index (0-9) を Showdown! プロトコル準拠のコマンド文字列に変換します。
    (既存の action_helper.py の内容とほぼ同じはずですが、player を使うようにします)
    """
    # 利用可能な技を move.id でソートして一貫性を担保
    available_moves_sorted: List[Move] = sorted(
        battle.available_moves, key=lambda m: m.id
    )
    # 利用可能な交代先（登録順）
    available_switches_list: List[Pokemon] = battle.available_switches

    _, mapping = get_available_actions(battle) # 内部でソート済みリストを使っている前提

    if action_index not in mapping:
        # 無効なアクションインデックスの場合、フォールバックとしてデフォルト行動 (例: 最初の有効な技)
        # またはエラーを送出。ここではエラー。
        print(f"Warning: Action index {action_index} not in mapping. Available mapping: {mapping}")
        # Gym環境の規約上は、エラーではなく、何かしらの有効な行動を返すか、
        # エージェント側でマスクされた行動を選ばないようにするべき。
        # ここでエラーを出すのは、開発中のデバッグのため。
        # 実際に学習させる際は、マスクされた行動が来た場合の処理を再考。
        # (例：ペナルティを与えてエピソード終了、ランダムな有効行動を選択)
        # 今回はValueErrorのままにしておく
        raise ValueError(f"Invalid or unavailable action index: {action_index}. Available mapping: {mapping}")


    action_type, idx_in_subtype_list = mapping[action_index]

    order = None
    if action_type == "move":
        if idx_in_subtype_list < len(available_moves_sorted):
            move_to_use = available_moves_sorted[idx_in_subtype_list]
            order = player.create_order(move_to_use, terastallize=False)
        else:
            raise ValueError(f"Move index {idx_in_subtype_list} out of bounds for available_moves (len: {len(available_moves_sorted)})")
    elif action_type == "terastal":
        if not battle.can_tera:
            raise ValueError("Cannot terastallize now, but terastal action was chosen.")
        if idx_in_subtype_list < len(available_moves_sorted):
            move_to_use = available_moves_sorted[idx_in_subtype_list]
            order = player.create_order(move_to_use, terastallize=True)
        else:
            raise ValueError(f"Terastal move index {idx_in_subtype_list} out of bounds for available_moves (len: {len(available_moves_sorted)})")
    elif action_type == "switch":
        if idx_in_subtype_list < len(available_switches_list):
            switch_to_pokemon = available_switches_list[idx_in_subtype_list]
            order = player.create_order(switch_to_pokemon)
        else:
            raise ValueError(f"Switch index {idx_in_subtype_list} out of bounds for available_switches (len: {len(available_switches_list)})")
    else:
        raise ValueError(f"Unknown action_type: {action_type}")

    return order