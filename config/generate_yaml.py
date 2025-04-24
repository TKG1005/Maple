"""
generate_yaml.py
ヒューマンリーダブルな Excel 台帳（状態空間表）を読み込み、
機械学習で使いやすい YAML 形式 (state_spec.yml) に変換するスクリプト。

前提
----
- 台帳ファイル名: 「状態空間表（仮）.xlsx」
  ※変更した場合は SOURCE_XLSX を書き換えてください
- 必須列: Category, Feature ID, Battle　経路, Type, Encoder, Default
  （無い列は自動で既定値を補完します）
"""

import pandas as pd        # 表計算を扱うライブラリ
import yaml                # YAML の読み書き
from pathlib import Path   # OS 依存しないパス操作
from poke_env.environment.move import Move
from poke_env.environment.pokemon_type import PokemonType


# ---------- 1. 入出力ファイルパス -----------------------------------------

SOURCE_XLSX = "state_feature_catalog_temp.csv"  # 入力: Excel 台帳
TARGET_YAML = "state_spec.yml"        # 出力: YAML ファイル

# ---------- 2. DataFrame → ネスト辞書 変換関数 ----------------------------

def build_yaml(df: pd.DataFrame) -> dict:
    spec: dict = {}

    for _, row in df.iterrows():
        cat = row.get("Category")
        feat = row.get("Feature ID")
        if pd.isna(cat) or pd.isna(feat):
            continue

        spec.setdefault(cat, {})

        entry = {
            "dtype": str(row.get("Type", "float32")).lower(),
            "battle_path": str(row.get("Battle　経路", "")).strip(),
            "encoder": str(row.get("Encoder", "identity")).strip(),
            "default": row.get("Default", 0),
        }

        # --- 新規追加（Range 列） ---
        range_val = row.get("Range")
        if isinstance(range_val, str) and "range" in range_val:
            # 例: 'range:[0,255] scale_to:[0,1]' を分解
            parts = [p.strip() for p in range_val.split(" ") if p]
            for p in parts:
                if p.startswith("range:"):
                    entry["range"] = eval(p.replace("range:", ""))
                elif p.startswith("scale_to:"):
                    entry["scale_to"] = eval(p.replace("scale_to:", ""))

        # --- 新規追加（classes 列） ---
        classes_val = row.get("classes")
        if isinstance(classes_val, str) and classes_val.strip().startswith("["):
            try:
                entry["classes"] = eval(classes_val.strip())
            except Exception:
                pass  # eval失敗時は無視

        spec[cat][feat] = entry

    return spec

# 取得したわざ一覧を固定順でソートする関数
def get_sorted_moves(pokemon) -> list[Move]:
    return sorted(pokemon.moves.values(), key=lambda m: m.id)[:4]

# ワンホットエンコーダの準備（タイプ）
TYPE_LIST = [None,  # unknown
    PokemonType.BUG, PokemonType.DARK, PokemonType.DRAGON,
    PokemonType.ELECTRIC, PokemonType.FAIRY, PokemonType.FIGHTING,
    PokemonType.FIRE, PokemonType.FLYING, PokemonType.GHOST,
    PokemonType.GRASS, PokemonType.GROUND, PokemonType.ICE,
    PokemonType.NORMAL, PokemonType.POISON, PokemonType.PSYCHIC,
    PokemonType.ROCK, PokemonType.STEEL, PokemonType.WATER,
    PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPE_LIST)}

def encode_type(t: PokemonType | None) -> list[int]:
    idx = TYPE_TO_IDX.get(t, 0)
    return [1 if i == idx else 0 for i in range(len(TYPE_LIST))]

def encode_class(cls: str | None) -> list[int]:
    class_list = ["physical", "special", "status", "unk"]
    idx = class_list.index(cls) if cls in class_list else 3
    return [1 if i == idx else 0 for i in range(len(class_list))]

def linear_scale(x: float, lo: float, hi: float, a: float = 0.0, b: float = 1.0) -> float:
    return (x - lo) * (b - a) / (hi - lo) + a

def extract_move_features(move: Move | None) -> list:
    if move is None:
        # 未判明スロット
        return encode_type(None) + [0.0] + encode_class(None) + [0.0]

    # 技の属性をエンコード
    type_vec = encode_type(move.type)
    power = linear_scale(move.base_power, 0, 255)
    class_vec = encode_class(move.damage_class)
    pp_frac = move.current_pp / move.max_pp if move.max_pp else 0.0

    return type_vec + [power] + class_vec + [pp_frac]
# ---------- 3. メイン処理 -------------------------------------------------

def main() -> None:
    """Excel 読み込み → dict 生成 → YAML 書き出し"""
    # 3‑A) Excel ファイルを読み込む（最初のシートを想定）
    df = pd.read_csv(SOURCE_XLSX)

    # 3‑B) DataFrame を辞書へ変換
    df_mvp = df[df["MVP"] == True]
    yaml_dict = build_yaml(df_mvp)

    # 3‑C) YAML ファイルとして保存（キー順は変更しない）
    with open(TARGET_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            yaml_dict,
            f,
            allow_unicode=True,   # 日本語もそのまま書き出す
            sort_keys=False       # 台帳の順序を保持
        )

    print(f"✅ YAML 生成完了: {TARGET_YAML}")

# ---------- 4. スクリプト実行エントリポイント ----------------------------

if __name__ == "__main__":
    main()
