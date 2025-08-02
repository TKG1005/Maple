<!--
  SHOWDOWN_MESSAGE_SPEC.md
  MapleShowdownCore メッセージ仕様書
  Python (IPCBattle) と Node.js MapleShowdownCore 間の IPC メッセージ仕様。
  Pokemon Showdown シミュレータ プロトコル (SIM-PROTOCOL.md) との対応を示す。
-->
# MapleShowdownCore メッセージ仕様書

## 1. 目的
このドキュメントは、Python (IPCBattle) と Node.js MapleShowdownCore 間の IPC メッセージ仕様を定義し、
Pokemon Showdown のテキストプロトコルとの対応付けを行います。

## 2. Showdown テキストプロトコル (参考)
以下は主要な Showdown テキストプロトコルメッセージの例です。
詳細は `pokemon-showdown/sim/SIM-PROTOCOL.md` を参照してください。

- `|player|PLAYER|USERNAME|AVATAR|RATING`
- `|teamsize|PLAYER|NUMBER`
- `|gametype|GAMETYPE`
- `|teampreview`, `|start`
- `|request|REQUEST_JSON`
- `|update|…`, `|sideupdate|PLAYER|…`
- `|win|PLAYER`, `|tie`

## 3. IPC JSON メッセージ

### 3.1 リクエスト (Python → Node.js MapleShowdownCore)

| type               | フィールド        | 型             | 必須 | 説明                                 |
|--------------------|-------------------|----------------|------|--------------------------------------|
| `create_battle`    | `battle_id`       | string         | ◯   | 一意なバトル ID                      |
|                    | `format`          | string         | ◯   | フォーマット ID (例: 'gen9randombattle') |
|                    | `players`         | Player[]       | ◯   | プレイヤー情報配列                   |
| `battle_command`   | `battle_id`       | string         | ◯   | バトル ID                            |
|                    | `command`         | string         | ◯   | Showdown コマンド (例: 'move 1')      |
| `get_battle_state` | `battle_id`       | string         | ◯   | バトル ID                            |
| `destroy_battle`   | `battle_id`       | string         | ◯   | バトル終了／破棄要求                 |

#### Player 型
```ts
interface Player {
  name: string;  // プレイヤー名
  team: string;  // team データ文字列 (Showdown 形式)
}
```

### 3.2 イベント通知 (Node.js MapleShowdownCore → Python) - 修正版

| type               | フィールド        | 型               | 必須 | 説明                             |
|--------------------|-------------------|------------------|------|----------------------------------|
| `battle_created`   | `battle_id`       | string           | ◯   | バトル作成完了                   |
|                    |                   |                  |      | **出力チャネル**: stdout         |
| `battle_update`    | `battle_id`       | string           | ◯   | バトル ID                        |
|                    |                   |                  |      | **出力チャネル**: stdout         |
|                    | `player_id`       | string           | ◯   | 対象プレイヤー ("p1" or "p2")    |
|                    | `log`             | string[]         | ◯   | プレイヤー固有のイベント配列     |
| `battle_end`       | `battle_id`       | string           | ◯   | バトル ID                        |
|                    |                   |                  |      | **出力チャネル**: stdout         |
|                    | `result`          | 'win' \| 'tie' | ◯   | 結果                             |
|                    | `winner?`         | string           | ×   | 勝者 (result='win' の場合)       |
| `error`            | `battle_id?`      | string?          | ×   | 対象バトル ID                    |
|                    |                   |                  |      | **出力チャネル**: stdout         |
|                    | `player_id?`      | string?          | ×   | 対象プレイヤー                   |
|                    | `message`         | string           | ◯   | エラーメッセージ                 |

#### ActiveInfo / SideInfo 型 (抜粋)
```ts
interface MoveChoice { move: number; target?: string; }

interface ActiveInfo {
  pokemon: string;      // 例: 'p1a'
  hp: number;           // 現在 HP (割合ではなく数値)
  maxhp: number;
  status?: string;      // 'par', 'brn' など
  choices: MoveChoice[]; // 選択可能行動
}

interface SideInfo {
  pokemon: TeamPokemon[]; // チーム全体
  // 他に残数、交代可能数など
}
```

## 4. 通信フロー例 - 修正版（不完全情報ゲーム対応）

### 4.1 バトル作成フロー
1. Python EnvPlayer A → MapleShowdownCore:
```json
{ "type":"create_battle", "battle_id":"b1", "format":"gen9randombattle", "players":[...] }
```
2. MapleShowdownCore → Python EnvPlayer A & B:
```json
{ "type":"battle_created", "battle_id":"b1" }
```

### 4.2 プレイヤー固有メッセージ配信
MapleShowdownCore は各プレイヤーに対して、生のShowdownプロトコル行（`>battle-…` や `|request|…` を含む各行）を**加工せずそのまま** `log` 配列に入れて送信します。
**出力チャネル**: これらの生テキスト行と JSON 形式のイベント通知は stdout にのみ書き出されます。
各EnvPlayer側では `player_id` タグで自分宛のメッセージを受け取り、`log` 内のすべての文字列をそのまま分割せずに `parse_message`（CustomBattle → Battle）に渡すことで、poke-env の標準処理を完全に再現します。

例: MapleShowdownCore → Python EnvPlayer A 向け JSON
```json
{
  "type": "battle_update",
  "battle_id": "b1",
  "player_id": "p1",
  "log": [
    ">battle-gen9bssregi-203804",
    "|request|{\"active\":[…],\"side\":{…},\"rqid\":36}",
    "|move|p1a|Glacial Lance|…",
    "…"
  ]
}
```

### 4.3 コマンド送信とメッセージフィルタリング
4. Python EnvPlayer A → MapleShowdownCore:
```json
{ "type":"battle_command", "battle_id":"b1", "command":"move 1" }
```
5. MapleShowdownCore → Python EnvPlayer A (A視点ログ):
```json
{ "type":"battle_update", "battle_id":"b1", "player_id":"p1", "log":["|move|p1a|Tackle|p2a","|turn|2",…] }
```
5. MapleShowdownCore → Python EnvPlayer B (B視点ログ):
```json
{ "type":"battle_update", "battle_id":"b1", "player_id":"p2", "log":["|move|p1a|Tackle|p2a","|turn|2",…] }
```

### 4.4 バトル終了
6. MapleShowdownCore → Python EnvPlayer A & B:
```json
{ "type":"battle_end", "battle_id":"b1", "result":"win", "winner":"p1" }
```

**重要**: 各EnvPlayerは自分宛て(`player_id`が一致)のメッセージのみ受信し、独立したBattleオブジェクトを更新します。

### 4.5 rqid保持要件
**MapleShowdownCore側の責務**:
- 生のShowdownログに含まれる`rqid`を必ず保持・転送
- `"log"`配列内のメッセージ（通常は最後）に`"rqid"`が含まれることを保証

**実装例**:
```json
// MapleShowdownCore → EnvPlayer
{
  "type": "battle_update",
  "battle_id": "b1", 
  "player_id": "p1",
  "log": [
    "|move|p1a|Tackle|p2a",
    "|turn|2",
    "|request|{\"requestType\":\"move\",\"rqid\":2,\"active\":[...]}"
  ]
}
```

これによりEnvPlayerは`battle.last_request`から正しく`rqid`を取得できます。

---
詳細は `SIM-PROTOCOL.md` の各メッセージ定義と照合してください。