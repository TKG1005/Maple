import csv, io, requests, textwrap
from bs4 import BeautifulSoup   # ← NEW

RAW = "https://raw.githubusercontent.com/veekun/pokedex/master/pokedex/data/csv/{}"
files = {
    "moves": "moves.csv",
    "move_names": "move_names.csv",
    "move_meta": "move_meta.csv",
    "move_flag_map": "move_flag_map.csv",
    "move_flags": "move_flags.csv",
    "move_meta_categories": "move_meta_categories.csv",
    "types": "types.csv",
    "move_damage_classes": "move_damage_classes.csv",
}

def fetch_csv(name):
    url = RAW.format(files[name])
    return list(csv.DictReader(io.StringIO(requests.get(url).text)))

data = {k: fetch_csv(k) for k in files}

# --- Yakkun から技説明を取得 -----------------------------------------------
YAKKUN_URL = "https://yakkun.com/sv/move_list.htm"
soup = BeautifulSoup(requests.get(YAKKUN_URL).content, "html.parser")

desc_map = {}  # {日本語名: 説明文}
for tr in soup.select("table tr")[1:]:          # 1行目はヘッダー
    tds = tr.find_all("td")
    if len(tds) < 8:                            # ヘッダーや区切り行を除外
        continue
    name_jp = tds[0].get_text(strip=True)
    effect  = tds[2].get_text(strip=True)       # 「効果」列（0:名前,1:タイプ,2:効果…）
    desc_map[name_jp] = effect
# ---------------------------------------------------------------------------

type_map = {r["id"]: r["identifier"] for r in data["types"]}
dmg_map  = {r["id"]: r["identifier"].title() for r in data["move_damage_classes"]}
flag_map = {r["id"]: r["identifier"] for r in data["move_flags"]}

flags = {}
for m in data["move_flag_map"]:
    flags.setdefault(m["move_id"], []).append(flag_map[m["move_flag_id"]])

names_jp = {r["move_id"]: r["name"] for r in data["move_names"] if r["local_language_id"]=="1"}
names_en = {r["move_id"]: r["name"] for r in data["move_names"] if r["local_language_id"]=="9"}

guard_moves = {
    "Protect","Detect","Endure","Spiky Shield","King's Shield",
    "Baneful Bunker","Obstruct","Silk Trap","Max Guard",
    "Wide Guard","Quick Guard","Crafty Shield","Mat Block"
}

rows=[]
for m in data["moves"]:
    mid=m["id"]; f=flags.get(mid,[]); meta=next((x for x in data["move_meta"] if x["move_id"]==mid),{})
    name_jp = names_jp.get(mid,"")
    row={
        "move_id": mid,
        "name": name_jp,
        "name_eng": names_en.get(mid,""),
        "category": dmg_map.get(m["damage_class_id"],""),
        "type": type_map.get(m["type_id"],"none"),
        "power": int(m["power"] or 0),
        "accuracy": float(m["accuracy"])/100 if m["accuracy"] else 1.0,
        "pp": int(m["pp"] or 0),
        "priority": int(m["priority"] or 0),
        "crit_stage": int(meta.get("crit_rate",0)),
        "ohko_flag": 1 if meta.get("meta_category_id")=="9" else 0,
        "contact_flag": int("contact" in f),
        "sound_flag": int("sound" in f),
        "multi_hit_min": int(meta.get("min_hits") or 1),
        "multi_hit_max": int(meta.get("max_hits") or 1),
        "recoil_ratio": round(max(-int(meta.get("drain",0)),0)/100,2),
        "recharge_turn": int("recharge" in f),
        "contenus_turn": int(int(meta.get("max_turns") or 0)>1),
        "charging_turn": int("charge" in f),
        "healing_ratio": round(int(meta.get("healing",0))/100,2),
        "protectable": int("protect" in f),
        "substitutable": int("sound" not in f),
        "guard_move_flag": int(names_en.get(mid,"") in guard_moves),
        # --- Yakkun 由来の説明文を使用 -------------------------------------
        "base_desc": desc_map.get(name_jp, "")
    }
    rows.append(row)

with open("moves.csv","w",newline="",encoding="utf-8") as fp:
    writer=csv.DictWriter(fp,fieldnames=row.keys())
    writer.writeheader(); writer.writerows(rows)

print("moves.csv generated with",len(rows),"rows")