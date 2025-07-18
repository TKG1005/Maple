#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pokémon SV move list scraper
取得元: https://yakkun.com/sv/move_list.htm  (日本語)
        https://yakkun.com/sv/move_en.htm    (英語名)
出力  : moves.csv  (UTF‑8 BOM 無し)
"""

import csv
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

JP_URL = "https://yakkun.com/sv/move_list.htm"
EN_URL = "https://yakkun.com/sv/move_en.htm"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# 18タイプ一覧（要件順）
TYPES = [
    "none","bug","dark","dragon","electric","fairy","fighting","fire","flying",
    "ghost","grass","ground","ice","normal","poison","psychic","rock","steel",
    "stellar","three_question_marks","water"
]

# 変換用辞書
CATEGORY_MAP = {"物理": "Physical", "特殊": "Special", "変化": "Status"}

def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def parse_jp_table(html: str):
    soup = BeautifulSoup(html, "html.parser")
    # 技テーブルはページ内で最初に「技リスト」という文字列が入る table
    table = next(t for t in soup.select("table") if "技リスト" in t.text)
    rows = []
    trs = table.select("tr")[2:]        # 1行目: ヘッダ, 2行目: サブヘッダ
    for tr in trs:
        tds = [td.get_text(strip=True) for td in tr.select("td")]
        if len(tds) < 8:          # 説明行(対象/効果)はスキップ
            continue
        name, type_jp, cat_jp, power, acc, pp, contact, protect = tds[:8]
        rows.append({
            "name": name,
            "type_jp": type_jp,
            "category_jp": cat_jp,
            "power": power if power != "-" else "0",
            "accuracy": acc.replace("—", "1.0").replace("-", "1.0"),
            "pp": pp,
            "contact_flag": "1" if "○" in contact else "0",
            "protectable": "1" if "○" in protect else "0",
        })
    return rows

def parse_en_table(html: str):
    soup = BeautifulSoup(html, "html.parser")
    table = next(t for t in soup.select("table") if "技リスト" in t.text or "Move List" in t.text)
    en_map = {}
    trs = table.select("tr")[2:]
    for tr in trs:
        tds = [td.get_text(strip=True) for td in tr.select("td")]
        if len(tds) >= 2:
            jp_name, en_name = tds[:2]
            en_map[jp_name] = en_name
    return en_map

def normalize_type(type_jp: str) -> str:
    jp_to_en = {
        "むし":"bug","あく":"dark","ドラゴン":"dragon","でんき":"electric","フェアリー":"fairy",
        "かくとう":"fighting","ほのお":"fire","ひこう":"flying","ゴースト":"ghost","くさ":"grass",
        "じめん":"ground","こおり":"ice","ノーマル":"normal","どく":"poison","エスパー":"psychic",
        "いわ":"rock","はがね":"steel","みず":"water","テラスタル":"stellar","???":"three_question_marks",
        "なし":"none"
    }
    return jp_to_en.get(type_jp, "none")

def main():
    print("Fetching Japanese move list...")
    jp_html = fetch_html(JP_URL)
    print("Fetching English move list...")
    en_html = fetch_html(EN_URL)

    print("Parsing tables...")
    jp_rows = parse_jp_table(jp_html)
    en_map = parse_en_table(en_html)

    # CSV カラム（要件順）
    cols = [
        "move_id","name","name_eng","category","type","power","accuracy","pp","priority",
        "crit_stage","ohko_flag","contact_flag","sound_flag","multi_hit_min","multi_hit_max",
        "recoil_ratio","recharge_turn","charging_turn","healing_ratio","protectable",
        "substitutable","guard_move_flag","base_desc"
    ]

    csv_rows = []
    for idx, row in enumerate(jp_rows, 1):
        # 基本情報
        name = row["name"]
        name_eng = en_map.get(name, "")
        category = CATEGORY_MAP.get(row["category_jp"], "Status")
        type_en = normalize_type(row["type_jp"])

        # 命中率を 0-1 float へ
        acc = row["accuracy"].replace("%", "")
        accuracy = str(float(acc) / 100) if acc not in ("1.0","—","--") else "1.0"

        # OHKO 判定 (名前に「じわれ」「ハサミギロチン」など含む簡易判定)
        ohko_flag = "1" if re.search(r"じわれ|ハサミギロチン|つのドリル|ぜったいれいど", name) else "0"

        # ひとまず取得できない項目は 0 / 空文字 で埋める
        csv_rows.append([
            idx,                   # move_id
            name,                  # name (JP)
            name_eng,              # name_eng
            category,              # category
            type_en,               # type
            row["power"],          # power
            accuracy,              # accuracy
            row["pp"],             # pp
            0,                     # priority (未取得)
            0,                     # crit_stage
            ohko_flag,             # ohko_flag
            row["contact_flag"],   # contact_flag
            0,                     # sound_flag
            1,                     # multi_hit_min (単発=1)
            1,                     # multi_hit_max
            0,                     # recoil_ratio
            0,                     # recharge_turn
            0,                     # charging_turn
            0,                     # healing_ratio
            row["protectable"],    # protectable
            1,                     # substitutable (仮)
            0,                     # guard_move_flag
            ""                     # base_desc (未取得)
        ])

    out_path = Path("moves.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(csv_rows)

    print(f"\nGenerated {out_path} with {len(csv_rows)} rows.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)