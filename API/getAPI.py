import json
import urllib.request
import sqlite3
from contextlib import closing

### 設定
season = 9
battle_rule = 1 # シングルなら1 ダブルなら2
dbname = 'single.db'

### ランクマの情報を取得
season_id = 10000 + season * 10 + battle_rule
user_agent = 'Mozilla/5.0 (Linux; Android 8.0; Pixel 2 Build/OPD3.170816.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Mobile Safari/537.36'

url = 'https://api.battle.pokemon-home.com/cbd/competition/rankmatch/list'
headers = {
    'User-Agent': user_agent,
    'countrycode': '304',
    'authorization': 'Bearer',
    'langcode': '1',
    'accept': 'application/json, text/javascript, */*; q=0.01',
    'content-type': 'application/json',
}
data = {
    'soft': 'Sw',
}

req = urllib.request.Request(url, json.dumps(data).encode(), headers)
body = ''
with urllib.request.urlopen(req) as res:
    body = json.load(res)

detail = body['list'][str(season)][str(season_id)]
ts2 = detail['ts2']
rst = detail['rst']

### 図鑑情報の読み込み
pokedex = ''
#  https://resource.pokemon-home.com/battledata/js/bundle.js の図鑑と持ち物情報を抜き出してjsonにしたものを読み込む
with open('./bundle.json', 'r') as json_open:
    pokedex = json.load(json_open)

### ポケモン情報の取得（今回はポケモンのわざともちものととくせいとせいかくの採用率を取得する）
headers = {
    'User-Agent': user_agent,
    'countrycode': '304',
    'authorization': 'Bearer',
    'langcode': '1',
    'accept': 'application/json, text/javascript, */*; q=0.01',
}

with closing(sqlite3.connect(dbname)) as conn:
    c = conn.cursor()

    ### テーブル作成
    c.execute( '''create table waza (name text, waza text, adoption_rate real)''')
    c.execute( '''create table item (name text, item text, adoption_rate real)''')
    c.execute('''create table tokusei (name text, tokusei text, adoption_rate real)''')
    c.execute('''create table seikaku (name text, seikaku text, adoption_rate real)''')

    for i in range(1, 6):
        url = f'https://resource.pokemon-home.com/battledata/ranking/{season_id}/{rst}/{ts2}/pdetail-{i}'
        req = urllib.request.Request(url, headers=headers)
        pdetail = ''
        with urllib.request.urlopen(req) as res:
            pdetail = json.load(res)

        for pokenum in pdetail.keys():
            for p_detail_id in pdetail[pokenum].keys():
                name = pokedex['poke'][int(pokenum) -1]
                if p_detail_id != '0': # 0以外はフォルム・性別・リージョンetc違いなので分けて扱う
                    name = name + p_detail_id
                for pokewaza in pdetail[pokenum][p_detail_id]['temoti']['waza']:
                    sql = 'insert into waza (name, waza, adoption_rate) values (?,?,?)'
                    waza = (name, pokedex['waza'][pokewaza['id']], pokewaza['val'])
                    c.execute(sql, waza)
                for pokeitem in pdetail[pokenum][p_detail_id]["temoti"]["motimono"]:
                    sql = 'insert into item (name, item, adoption_rate) values (?,?,?)'
                    item = (name, pokedex['item'][pokeitem['id']], pokeitem['val'])
                    c.execute(sql, item)
                for poketokusei in pdetail[pokenum][p_detail_id]['temoti']['tokusei']:
                    sql = 'insert into tokusei (name, tokusei, adoption_rate) values (?,?,?)'
                    tokusei = (name, pokedex["tokusei"][poketokusei["id"]],  poketokusei["val"])
                    c.execute(sql, tokusei)
                for pokeseikaku in pdetail[pokenum][p_detail_id]['temoti']['seikaku']:
                    sql = 'insert into seikaku (name, seikaku, adoption_rate) values (?,?,?)'
                    seikaku = (name, pokedex["seikaku"][pokeseikaku["id"]],  pokeseikaku["val"])
                    c.execute(sql, seikaku)

    conn.commit()