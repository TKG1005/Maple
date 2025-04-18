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

# ---------- 1. 入出力ファイルパス -----------------------------------------

SOURCE_XLSX = "state_feature_catalog_temp.csv"  # 入力: Excel 台帳
TARGET_YAML = "state_spec.yml"        # 出力: YAML ファイル

# ---------- 2. DataFrame → ネスト辞書 変換関数 ----------------------------

def build_yaml(df: pd.DataFrame) -> dict:
    """
    行単位で DataFrame を走査し、
    {Category: {Feature ID: メタ情報…}} の辞書を生成する。

    Parameters
    ----------
    df : pandas.DataFrame
        読み込んだ Excel シート

    Returns
    -------
    dict
        YAML へそのまま safe_dump できるネスト辞書
    """
    spec: dict = {}

    # → DataFrame を 1 行ずつ処理
    for _, row in df.iterrows():

        # 2‑A) カテゴリとフィーチャ名を取得（空ならスキップ）
        cat  = row.get("Category")
        feat = row.get("Feature ID")
        if pd.isna(cat) or pd.isna(feat):
            continue  # 必須列が欠けている行は無視

        # 2‑B) カテゴリ辞書が無ければ作成
        spec.setdefault(cat, {})

        # 2‑C) 個々のフィーチャに対するメタ情報を構築
        entry = {
            # dtype: 空欄なら float32 を既定に
            "dtype": str(row.get("Type", "float32")).lower(),

            # battle_path: poke‑env 内の取得パス
            "battle_path": str(row.get("Battle　経路", "")).strip(),

            # encoder: 未指定なら identity
            "encoder": str(row.get("Encoder", "identity")).strip(),

            # default: NaN の場合は 0 で埋める
            "default": row.get("Default", 0)
        }

        # 2‑D) ネスト辞書へ登録
        spec[cat][feat] = entry

    return spec

# ---------- 3. メイン処理 -------------------------------------------------

def main() -> None:
    """Excel 読み込み → dict 生成 → YAML 書き出し"""
    # 3‑A) Excel ファイルを読み込む（最初のシートを想定）
    df = pd.read_excel(SOURCE_XLSX)

    # 3‑B) DataFrame を辞書へ変換
    yaml_dict = build_yaml(df)

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
    main
