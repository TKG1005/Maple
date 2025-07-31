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

### 3.2 イベント通知 (Node.js → Python MapleShowdownCore)

| type               | フィールド        | 型               | 必須 | 説明                             |
|--------------------|-------------------|------------------|------|----------------------------------|
| `battle_created`   | `battle_id`       | string           | ◯   | バトル作成完了                   |
| `choice_request`   | `battle_id`       | string           | ◯   | リクエスト対象バトル ID          |
|                    | `rqid`            | number           | ◯   | リクエスト ID                    |
|                    | `active`          | ActiveInfo       | ◯   | アクティブポケモン情報           |
|                    | `side`            | SideInfo         | ◯   | チーム全体情報                   |
|                    | `forceSwitch`     | boolean          | ×   | 強制交代フラグ                   |
| `battle_update`    | `battle_id`       | string           | ◯   | バトル ID                        |
|                    | `log`             | string[]         | ◯   | Showdown テキストイベント配列    |
| `battle_end`       | `battle_id`       | string           | ◯   | バトル ID                        |
|                    | `result`          | 'win' \| 'tie' | ◯   | 結果                             |
|                    | `winner?`         | string           | ×   | 勝者 (result='win' の場合)       |
| `error`            | `battle_id?`      | string?          | ×   | 対象バトル ID                    |
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

## 4. 通信フロー例

1. Python → Node.js:
```json
{ "type":"create_battle", "battle_id":"b1", "format":"gen9randombattle", "players":[...] }
```
2. Node.js → Python:
```json
{ "type":"battle_created", "battle_id":"b1" }
```
3. Node.js → Python (リクエスト):
```json
{ "type":"choice_request", "battle_id":"b1", "rqid":1, "active":{...}, "side":{...} }
```
4. Python → Node.js:
```json
{ "type":"battle_command", "battle_id":"b1", "command":"move 1" }
```
5. Node.js → Python (更新):
```json
{ "type":"battle_update", "battle_id":"b1", "log":["|move|p1a|Tackle|p2a","|turn|2",…] }
```
6. Node.js → Python (終了):
```json
{ "type":"battle_end", "battle_id":"b1", "result":"win", "winner":"p1" }
```

---
詳細は `SIM-PROTOCOL.md` の各メッセージ定義と照合してください。