# src/action/action_masker.py

import numpy as np

def generate_action_mask(available_moves, available_switches, can_terastallize):
    """
    固定長10の行動空間（通常技4、テラスタル技4、交代2）に対するマスクベクトルを生成する。

    Returns: np.array([1, 1, ..., -1])（長さ10）
    """

    mask = np.full(10, 0)  # 全部 -1 で初期化

    # 通常技スロット 0〜3
    for i in range(min(4, len(available_moves))):
        if available_moves[i]:
            mask[i] = 1

    # テラスタル技スロット 4〜7
    if can_terastallize:
        for i in range(min(4, len(available_moves))):
            if available_moves[i]:
                mask[i + 4] = 1

    # 交代スロット 8〜9
    for i in range(min(2, len(available_switches))):
        if available_switches[i]:
            mask[i + 8] = 1

    return mask
