from typing import List, Tuple, Dict, Union
import numpy as np
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.player.player import Player
from src.action.action_masker import generate_action_mask

ActionObject = Union[Move, Pokemon]


def get_available_actions(battle: Battle) -> Tuple[np.ndarray, Dict[int, Tuple[str, int]]]:
    """
    Battle オブジェクトから現在可能な行動をマスク付きインデックス形式で取得します。

    Returns
    -------
    mask : np.ndarray
        固定長10のマスクベクトル (1: 実行可, 0: 実行不可)
    mapping : Dict[int, Tuple[str, int]]
        action_index -> (action_type, idx)
        action_type は 'move', 'terastal', 'switch' のいずれか
        idx はソート済み available_moves または available_switches のインデックス
    """
    # 利用可能な技を move.id でソートして一貫性を担保
    available_moves: List[Move] = sorted(
        battle.available_moves, key=lambda m: m.id
    )
    # 利用可能な交代先（登録順）
    available_switches: List[Pokemon] = battle.available_switches
    # テラスタル可否 (can_tera が None でなければ可能)
    can_terastallize: bool = battle.can_tera is not None

    # ブールリスト化
    moves_mask = [move.current_pp > 0 for move in available_moves]
    switches_mask = [True] * len(available_switches)

    # マスク生成
    mask = generate_action_mask(moves_mask, switches_mask, can_terastallize)

    # インデックス→アクション種別のマッピング
    mapping: Dict[int, Tuple[str, int]] = {}
    # 通常技 0-3
    for i in range(min(4, len(moves_mask))):
        if moves_mask[i]:
            mapping[i] = ("move", i)
    # テラスタル技 4-7
    if can_terastallize:
        for i in range(min(4, len(moves_mask))):
            if moves_mask[i]:
                mapping[i + 4] = ("terastal", i)
    # 交代 8-9
    for i in range(min(2, len(switches_mask))):
        if switches_mask[i]:
            mapping[i + 8] = ("switch", i)

    return mask, mapping


def action_index_to_order(player: Player, battle: Battle, action_index: int) -> str:
    """
    選択された action_index (0-9) を Showdown! プロトコル準拠のコマンド文字列に変換します。

    Parameters
    ----------
    player : Player
        create_order を呼び出す主体 (self)
    battle : Battle
        現在のバトルインスタンス
    action_index : int
        get_available_actions で得たインデックス

    Returns
    -------
    str
        Showdown プロトコルの命令文字列 (例: "move 1", "move 2 terastallize", "switch 1")
    """
    _, mapping = get_available_actions(battle)

    if action_index not in mapping:
        raise ValueError(f"Invalid or unavailable action index: {action_index}")

    action_type, idx = mapping[action_index]

    if action_type == "move":
        move: Move = battle.available_moves[idx]
        return player.create_order(move)
    elif action_type == "terastal":
        move: Move = battle.available_moves[idx]
        return player.create_order(move, terastallize=True)
    elif action_type == "switch":
        pokemon: Pokemon = battle.available_switches[idx]
        return player.create_order(pokemon)
    else:
        raise ValueError(f"Unsupported action type: {action_type}")
