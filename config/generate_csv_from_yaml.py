"""
generate_csv_from_yaml.py
state_spec.yml から state_feature_catalog.csv に逆変換するユーティリティ。

前提
----
- 入力: config/state_spec.yml（本リポジトリで使用しているフォーマット）
- 出力: config/state_feature_catalog.csv（generate_yaml.py が読む想定の列）

注意
----
- CSV 側の列は最小集合（Category, Feature ID, Battle　経路, Type, Encoder, Default, Range, classes, MVP）のみを出力。
- MVP 列は YAML 側に情報が無いため、すべて True（1）で出力します。
- Range は YAML の `range` と `scale_to` を `"range:[lo,hi] scale_to:[a,b]"` 形式にまとめます。
- classes は Python リスト表現（例: "['none','brn',...]"）の文字列として出力します。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


CURRENT_DIR = Path(__file__).resolve().parent
SOURCE_YAML = CURRENT_DIR / "state_spec.yml"
TARGET_CSV = CURRENT_DIR / "state_feature_catalog.csv"


def _format_range_token(values: List[float | int]) -> str:
    """Return token like "[0,1]" without spaces for stable parsing in generate_yaml.py."""
    return "[" + ",".join(str(v) for v in values) + "]"


def build_dataframe(spec: Dict[str, Dict[str, Dict[str, Any]]]) -> pd.DataFrame:
    """Build a DataFrame compatible with generate_yaml.py from YAML spec mapping.

    Columns:
    - Category
    - Feature ID
    - Battle　経路
    - Type
    - Encoder
    - Default
    - Range (e.g., "range:[0,255] scale_to:[0,1]")
    - classes (e.g., "['none','brn',...]")
    - MVP (bool/int) — always True(1)
    """
    rows: List[Dict[str, Any]] = []

    # spec: { category: { feature: { ...entry... } } }
    for category, features in spec.items():
        if not isinstance(features, dict):
            continue
        for feature, entry in features.items():
            if not isinstance(entry, dict):
                continue

            dtype = entry.get("dtype", "")
            battle_path = entry.get("battle_path", "")
            encoder = entry.get("encoder", "")
            default = entry.get("default", "")

            # Range column formatting
            rng = entry.get("range")
            scale_to = entry.get("scale_to")
            range_cell = ""
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                range_cell = f"range:{_format_range_token(list(rng))}"
            if isinstance(scale_to, (list, tuple)) and len(scale_to) == 2:
                tok = f"scale_to:{_format_range_token(list(scale_to))}"
                range_cell = f"{range_cell} {tok}".strip()

            # classes column formatting -> Python list literal of strings
            classes_list = entry.get("classes")
            classes_cell = ""
            if isinstance(classes_list, list) and classes_list:
                # 文字列へ正規化し、クォート付きで連結
                def _q(x: Any) -> str:
                    return "'" + str(x) + "'"

                classes_cell = "[" + ",".join(_q(x) for x in classes_list) + "]"

            # Default は CSV では文字列として扱う。YAML が数値等でも str() へ。
            default_cell = str(default)

            row = {
                "Category": category,
                "Feature ID": feature,
                "Battle　経路": battle_path,
                "Type": str(dtype),
                "Encoder": str(encoder),
                "Default": default_cell,
                "Range": range_cell,
                "classes": classes_cell,
                "MVP": 1,  # YAML からは判別不可のため既定で True
            }
            rows.append(row)

    # 固定の列順
    columns = [
        "Category",
        "Feature ID",
        "Battle　経路",
        "Type",
        "Encoder",
        "Default",
        "Range",
        "classes",
        "MVP",
    ]

    return pd.DataFrame(rows, columns=columns)


def main(yaml_path: Path | None = None, csv_path: Path | None = None) -> None:
    yaml_path = yaml_path or SOURCE_YAML
    csv_path = csv_path or TARGET_CSV

    with open(yaml_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}
        if not isinstance(spec, dict):
            raise TypeError("state_spec.yml must be a mapping at top-level")

    df = build_dataframe(spec)
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV 生成完了: {csv_path}")


if __name__ == "__main__":
    main()

