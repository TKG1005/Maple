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

### メッセージ形式の統一
- すべてのstdout出力はJSON形式で統一
- IPCプロトコルメッセージ：`type`フィールドで識別（`battle_created`, `error`など）
- Showdownプロトコルメッセージ：`{"type": "protocol", "data": "..."}`形式でラップ
- エラーメッセージはstderrに出力、IPCエラーはJSON形式でstdoutにも出力

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

### 3.2 イベント通知 (Node.js MapleShowdownCore → Python)

#### IPCプロトコルメッセージ（識別子付き）

| type               | フィールド        | 型               | 必須 | 説明                             |
|--------------------|-------------------|------------------|------|----------------------------------|
| `battle_created`   | `battle_id`       | string           | ◯   | バトル作成完了                   |
| `battle_update`    | `battle_id`       | string           | ◯   | バトル ID                        |
|                    | `player_id`       | string           | ◯   | 対象プレイヤー ("p1" or "p2")    |
|                    | `log`             | string[]         | ◯   | プレイヤー固有のイベント配列     |
| `battle_end`       | `battle_id`       | string           | ◯   | バトル ID                        |
|                    | `result`          | 'win' \| 'tie' | ◯   | 結果                             |
|                    | `winner?`         | string           | ×   | 勝者 (result='win' の場合)       |
| `error`            | `error_type`      | string           | ◯   | エラータイプ                     |
|                    | `error_message`   | string           | ◯   | エラーメッセージ                 |
|                    | `context`         | object           | ×   | エラーコンテキスト               |

#### Showdownプロトコルメッセージ（JSON形式でラップ）

| type               | フィールド        | 型               | 必須 | 説明                             |
|--------------------|-------------------|------------------|------|----------------------------------|
| `protocol`         | `battle_id`       | string           | ◯   | バトル ID                        |
|                    | `player_id`       | string           | ×   | 対象プレイヤー（省略時は両方）   |
|                    | `data`            | string           | ◯   | Showdownオリジナルメッセージ     |

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
MapleShowdownCore はWebSocketモードと同じ形式でメッセージを送信します：
- **IPCプロトコルメッセージ**: `{"type": "battle_created", ...}`形式で送信
- **Showdownプロトコルメッセージ**: `{"type": "protocol", "data": "..."}`形式でラップ
- **メッセージ形式**: 複数行を改行（`\n`）で結合した1つの長い文字列
- **最初の行**: 必ず`>battle-format-id`形式のバトルタグ
- **出力チャネル**: すべてのメッセージはstdoutにJSON形式で出力、エラーはstderrに出力

例: MapleShowdownCore → Python 向け JSON
```json
// Showdownプロトコルメッセージ（WebSocket形式と同じ）
{
  "type": "protocol",
  "battle_id": "b1",
  "player_id": "p1",
  "data": ">battle-gen9bssregi-203804\n|init|battle\n|title|Player1 vs. Player2\n|j|☆Player1\n|request|{\"active\":[…],\"side\":{…},\"rqid\":36}"
}

// IPCプロトコルメッセージ
{
  "type": "battle_created",
  "battle_id": "b1",
  "success": true
}
```

### 4.3 コマンド送信とメッセージフィルタリング
4. Python EnvPlayer A → MapleShowdownCore:
```json
{ "type":"battle_command", "battle_id":"b1", "command":"move 1" }
```
5. MapleShowdownCore → Python (Showdownメッセージ):
```json
// プレイヤー1向け（複数行を改行で結合）
{ 
  "type":"protocol", 
  "battle_id":"b1", 
  "player_id":"p1", 
  "data":">battle-gen9bssregi-b1\n|move|p1a|Tackle|p2a\n|-damage|p2a|80/100\n|turn|2\n|request|{\"active\":[...],\"rqid\":2}"
}

// プレイヤー2向け（複数行を改行で結合）
{ 
  "type":"protocol", 
  "battle_id":"b1", 
  "player_id":"p2", 
  "data":">battle-gen9bssregi-b1\n|move|p1a|Tackle|p2a\n|-damage|p2a|80/100\n|turn|2\n|request|{\"active\":[...],\"rqid\":2}" 
}
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
- `|request|`メッセージに`rqid`が含まれることを保証（存在しない場合は追加）
- メッセージは必ずバトルタグから始まる改行区切りの文字列として送信

**実装例**:
```json
// MapleShowdownCore → EnvPlayer (WebSocket形式と同じ)
{
  "type": "protocol",
  "battle_id": "b1", 
  "player_id": "p1",
  "data": ">battle-gen9bssregi-b1\n|move|p1a|Tackle|p2a\n|turn|2\n|request|{\"requestType\":\"move\",\"rqid\":2,\"active\":[...]}"
}
```

これによりEnvPlayerは標準のpoke-env処理パイプラインでメッセージを処理でき、`battle.last_request`から正しく`rqid`を取得できます。

---
詳細は `SIM-PROTOCOL.md` の各メッセージ定義と照合してください。