# src/action/action_masker.py

import numpy as np

def generate_action_mask(
    available_moves,
    available_switches,
    can_tera,
    force_switch: bool = False,   # ★追加
):
    """
    固定長10（通常技4・テラス技4・交代2）のマスクを返す。
    force_switch=True のときは技／テラスを 0 にし、交代だけ 1 にする。
    """
    mask = np.zeros(10, dtype=np.int8)

    # --- 技スロット --------------------------------------------------------
    if not force_switch:                       # ★ここがポイント
        for i, _ in enumerate(available_moves[:4]):
            mask[i] = 1

        # --- テラスタル技 -------------------------------------------------
        if can_tera:
            for i, _ in enumerate(available_moves[:4]):
                mask[4 + i] = 1

    # --- 交代スロット ------------------------------------------------------
    for i, _ in enumerate(available_switches[:2]):
        mask[8 + i] = 1

    return mask