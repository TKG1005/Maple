# Pokemon Showdown メッセージ仕様

## 概要

Pokemon Showdown WebSocket プロトコルの詳細仕様です。IPC 版でもこのプロトコル形式を維持することで、既存の poke-env との互換性を保っています。

---

## WebSocket 接続

**接続URL:**
- プロダクション: `wss://sim3.psim.us/showdown/websocket`
- ローカル開発: `ws://localhost:8000/showdown/websocket`

**プロトコル:** Pokemon Showdown は SockJS 互換レイヤーを使用しますが、直接 WebSocket 接続も可能です。

---

## メッセージ形式

### サーバーからクライアントへ（受信）

```
>ROOMID
MESSAGE
MESSAGE
...
```

- `>ROOMID` はルーム識別子（バトルの場合は `>battle-<format>-<id>`）
- 各 `MESSAGE` は個別のプロトコル行
- ロビー/グローバルメッセージでは `>ROOMID` は省略される

### クライアントからサーバーへ（送信）

```
ROOMID|TEXT
```

- `ROOMID` はルーム識別子（グローバルコマンドでは空白）
- `TEXT` には改行を含めることができる（個別メッセージとして扱われる）

---

## プロトコルメッセージ

### 基本形式

```
|TYPE|DATA|DATA|...
```

各メッセージは `|` で区切られたフィールドを持ちます。

### バトル初期化メッセージ

```
|player|p1|Username|60|1200
|player|p2|Username|113|1300
|teamsize|p1|4
|teamsize|p2|5
|gametype|singles
|gen|9
|tier|[Gen 9] Random Battle
|rule|Species Clause: Limit one of each Pokémon
|clearpoke
|poke|p1|Pikachu, L59, F|item
|poke|p2|Garchomp, M|item
|teampreview
|start
```

### バトル進行メッセージ

```
|turn|1
|move|p1a: Pikachu|Thunderbolt|p2a: Garchomp
|-damage|p2a: Garchomp|85/100
|-supereffective|p2a: Garchomp
|upkeep
```

### リクエストメッセージ（プレイヤー判断）

```
|request|{"active":[{"moves":[{"move":"Thunderbolt","id":"thunderbolt","pp":24,"maxpp":24,"target":"normal","disabled":false}]}],"side":{"name":"Player1","id":"p1","pokemon":[...]},"rqid":3}
```

---

## 主要アクションメッセージ

### メジャーアクション

- `|move|POKEMON|MOVE|TARGET` - ポケモンが技を使用
- `|switch|POKEMON|DETAILS|HP STATUS` - ポケモンが交代
- `|drag|POKEMON|DETAILS|HP STATUS` - 強制交代（吹き飛ばし等）
- `|cant|POKEMON|REASON|MOVE` - ポケモンが行動不能
- `|faint|POKEMON` - ポケモンがひんし

### マイナーアクション

- `|-damage|POKEMON|HP STATUS` - HPダメージ
- `|-heal|POKEMON|HP STATUS` - HP回復
- `|-status|POKEMON|STATUS` - 状態異常付与
- `|-boost|POKEMON|STAT|AMOUNT` - 能力上昇
- `|-fail|POKEMON|ACTION` - 行動失敗
- `|-immune|POKEMON` - 無効化
- `|-miss|SOURCE|TARGET` - 攻撃がはずれ
- `|-crit|POKEMON` - 急所に当たった
- `|-supereffective|POKEMON` - 効果は抜群
- `|-resisted|POKEMON` - 効果は今ひとつ

### ルームメッセージ

- `|init|ROOMTYPE` - ルーム初期化
- `|title|TITLE` - ルームタイトル
- `|users|USERLIST` - ユーザーリスト
- `|j|USER` - ユーザー参加
- `|l|USER` - ユーザー退出
- `|c|USER|MESSAGE` - チャットメッセージ

---

## ポケモン識別形式

### ポケモンID

```
POSITION:NAME
```

- `POSITION` = `PLAYER` + ポジション文字（例: `p1a`, `p2b`）
- `NAME` = ニックネームまたは種族名

### 詳細文字列

```
SPECIES, L##, GENDER, STATUS
```

例:
- `Garchomp, L50, M, shiny`
- `Pikachu` (レベル100がデフォルト)

---

## バトルコマンド（クライアントからサーバーへ）

### 技・交代コマンド

```
/choose move 1          # 1番目の技を使用
/choose move Thunderbolt # 技名で指定
/choose switch 2        # 2番目のポケモンに交代
```

### チームプレビュー

```
/choose team 213456     # チーム順序の変更
```

### ダブルバトル形式

```
/choose move 1 1, switch 2  # 1番目のポケモンで1番目の相手を攻撃、2番目のポケモンは交代
```

---

## エラーハンドリング

```
|error|[Invalid choice] MESSAGE
|error|[Unavailable choice] MESSAGE
|request|REQUEST  # エラー後の更新されたリクエスト
```

---

## IPC vs WebSocket プロトコルの違い

### WebSocket モード（オンライン）

- Pokemon Showdown プロトコルを直接 WebSocket で送受信
- メッセージは生のテキスト文字列として送信
- 完全な認証が必要

### IPC モード（ローカル）

- Pokemon Showdown プロトコルを JSON でラップ
- メタデータ付きの JSON 形式:

```json
{
  "type": "protocol",
  "battle_id": "battle-123", 
  "data": ">battle-gen9randombattle-123\n|move|p1a: Pikachu|Thunderbolt|p2a: Garchomp"
}
```

---

## 実装上の注意点

1. **メッセージ分割**: メッセージは `|` で分割しますが、チャットメッセージには `|` が含まれる可能性があります
2. **バトルタグ**: メッセージは `>battle-FORMAT-ID` でバトルを識別します
3. **リクエストID**: バトルの判断には同期用の `rqid` が含まれます
4. **改行**: メッセージは改行で区切られます
5. **JSON リクエスト**: 技・交代リクエストには完全なバトル状態の JSON が含まれます

このプロトコルは、デバッグ用の人間可読性と自動クライアント用の機械解析性の両方を考慮して設計されています。