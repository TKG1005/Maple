# Pokemon Showdown仕様準拠チェックリスト

## 🎯 最重要: 完全仕様準拠の実現

このプロジェクトの成功は**Pokemon Showdownサーバーの挙動を完璧に再現**できるかにかかっています。以下の仕様準拠が必須です。

## 📋 BattleStream API準拠

### ✅ Core API Usage
- [ ] **正確な初期化**: `const { BattleStream } = require('./dist/sim/battle-stream')`
- [ ] **適切なオプション**: `new BattleStream({ debug: false, noCatch: false, keepAlive: true })`
- [ ] **ストリーム継承**: `ObjectReadWriteStream<string>`の正確な実装

### ✅ Battle Initialization Sequence  
```javascript
// 必須: この順序での初期化
stream.write(`>start {"formatid":"gen9randombattle"}`);
stream.write(`>player p1 {"name":"Player1","team":null}`);  // random battle
stream.write(`>player p2 {"name":"Player2","team":null}`);
```

### ✅ Message Format Compliance
- [ ] **Update messages**: `update\nMESSAGES` (全プレイヤー・観戦者向け)
- [ ] **Side updates**: `sideupdate\nPLAYERID\nMESSAGES` (個別プレイヤー向け)  
- [ ] **End messages**: `end\nLOGDATA` (バトル終了)
- [ ] **Split messages**: `|split|PLAYERID` (秘匿・公開情報分離)

## 📋 SIM-PROTOCOL準拠

### ✅ Battle Initialization Messages
- [ ] `|player|p1|USERNAME|AVATAR|RATING`
- [ ] `|teamsize|p1|NUMBER`  
- [ ] `|gametype|singles|doubles|triples|multi|freeforall`
- [ ] `|gen|GENNUM` (1-9)
- [ ] `|tier|FORMATNAME`
- [ ] `|rule|RULE: DESCRIPTION` (複数回)
- [ ] `|clearpoke` + `|poke|PLAYER|DETAILS|ITEM` + `|teampreview`
- [ ] `|start`

### ✅ Major Battle Actions
- [ ] `|move|POKEMON|MOVE|TARGET` (技使用)
- [ ] `|switch|POKEMON|DETAILS|HP STATUS` (交代)
- [ ] `|drag|POKEMON|DETAILS|HP STATUS` (強制交代) 
- [ ] `|faint|POKEMON` (ひんし)
- [ ] `|cant|POKEMON|REASON|MOVE` (行動不能)

### ✅ Minor Battle Actions
- [ ] `|-damage|POKEMON|HP STATUS` 
- [ ] `|-heal|POKEMON|HP STATUS`
- [ ] `|-status|POKEMON|STATUS`
- [ ] `|-curestatus|POKEMON|STATUS`
- [ ] `|-boost|POKEMON|STAT|AMOUNT`
- [ ] `|-unboost|POKEMON|STAT|AMOUNT`
- [ ] `|-weather|WEATHER`
- [ ] `|-fieldstart|CONDITION`
- [ ] `|-fieldend|CONDITION`
- [ ] `|-sidestart|SIDE|CONDITION`
- [ ] `|-sideend|SIDE|CONDITION`

### ✅ Pokemon Identification System
- [ ] **Pokemon ID**: `POSITION: NAME` (例: `p1a: Pikachu`)
- [ ] **Position format**: `p1a`, `p1b` (シングル), `p1a`, `p1b`, `p2a`, `p2b` (ダブル)
- [ ] **Details format**: `SPECIES, L##, M/F, shiny, tera:TYPE`

## 📋 Choice Request System

### ✅ Request Format
```json
{
  "active": [
    {
      "moves": [
        {
          "move": "Thunder Shock",
          "id": "thundershock", 
          "pp": 30,
          "maxpp": 30,
          "target": "normal",
          "disabled": false
        }
      ]
    }
  ],
  "side": {
    "name": "Player1",
    "id": "p1", 
    "pokemon": [
      {
        "ident": "p1: Pikachu",
        "details": "Pikachu, L50, M",
        "condition": "150/150",
        "active": true,
        "stats": { "atk": 100, "def": 90, "spa": 120, "spd": 100, "spe": 200 },
        "moves": ["thundershock", "quickattack", "tailwhip", "growl"],
        "baseAbility": "static",
        "item": "",
        "ability": "static"
      }
    ]
  },
  "rqid": 1
}
```

### ✅ Choice Format
- [ ] **Move choices**: `move 1`, `move Thunder Shock`, `move 1 +1` (対象指定)
- [ ] **Switch choices**: `switch 2`, `switch Charmander`  
- [ ] **Team preview**: `team 123456`
- [ ] **Special modifiers**: `move 1 mega`, `move 1 zmove`, `move 1 max`

## 📋 Teams API準拠

### ✅ Team Generation
```javascript
const { Teams } = require('./dist/sim/teams');

// Random battle team generation
const team = Teams.generate('gen9randombattle');
const packedTeam = Teams.pack(team);

// Team validation
const validatedTeam = Teams.validate(team, 'gen9randombattle');
```

### ✅ Team Formats
- [ ] **Export format**: Human-readable (Teambuilder用)
- [ ] **JSON format**: `PokemonSet[]` (内部処理用)
- [ ] **Packed format**: 圧縮形式 (通信・保存用)

### ✅ Pokemon Set Structure
```json
{
  "name": "",
  "species": "Pikachu", 
  "gender": "M",
  "item": "Light Ball",
  "ability": "Static",
  "evs": {"hp": 0, "atk": 252, "def": 0, "spa": 252, "spd": 4, "spe": 0},
  "nature": "Modest",
  "ivs": {"hp": 31, "atk": 0, "def": 31, "spa": 31, "spd": 31, "spe": 31},
  "moves": ["Thunder Shock", "Quick Attack", "Tail Whip", "Growl"],
  "level": 50
}
```

## 📋 Critical Implementation Points

### 🚨 絶対に守るべき仕様
1. **メッセージ順序**: Showdownは厳密な順序でメッセージ送信
2. **HP表記**: `/100` (HP Percentage) vs `/48` (Exact HP)
3. **Gender表記**: `M`, `F`, または空文字（性別不明）
4. **Shiny表記**: `, shiny` 付与の正確なタイミング
5. **Tera表記**: Gen9では `, tera:TYPE` 必須

### 🚨 エラーが発生しやすい箇所
1. **Species名正規化**: "Pikachu" vs "pikachu" vs "PIKACHU"
2. **Move ID変換**: "Thunder Shock" → "thundershock"
3. **Request ID管理**: RQID重複によるdesync
4. **Choice validation**: 無効な行動の適切な拒否
5. **Team preview**: チーム順序の管理

## 📋 検証方法

### ✅ Comparison Testing
```bash
# WebSocketモードとIPCモードの完全比較
python test_showdown_compliance.py --mode websocket --episodes 10 --log battle_ws.log
python test_showdown_compliance.py --mode ipc --episodes 10 --log battle_ipc.log
diff battle_ws.log battle_ipc.log  # 差分は0であるべき
```

### ✅ Protocol Message Validation
```javascript
// 全メッセージタイプの受信確認
const requiredMessages = [
  '|player|', '|teamsize|', '|gametype|', '|gen|', '|tier|',
  '|clearpoke|', '|poke|', '|teampreview|', '|start|',
  '|turn|', '|move|', '|switch|', '|-damage|', '|win|'
];
```

### ✅ Battle Mechanics Verification
```python
# 既存WebSocketバトルとの結果一致性確認
def verify_battle_consistency():
    websocket_result = run_websocket_battle(seed=12345)
    ipc_result = run_ipc_battle(seed=12345) 
    assert websocket_result.winner == ipc_result.winner
    assert websocket_result.turn_count == ipc_result.turn_count
    assert websocket_result.final_hp == ipc_result.final_hp
```

## 🎯 成功判定基準

### Phase A完了基準
- [ ] BattleStream正常初期化
- [ ] Basic ping-pong IPC通信
- [ ] 1バトル完全実行（winner決定まで）

### Phase B完了基準  
- [ ] 全SIM-PROTOCOLメッセージ対応
- [ ] Choice request正確処理
- [ ] 10バトル連続成功実行

### 最終完了基準
- [ ] WebSocketモードとIPC mode で100%同一の結果
- [ ] 全Pokemon種族・技での正常動作
- [ ] 1000エピソード安定実行
- [ ] パフォーマンス目標達成（75%向上）

---

**重要**: このチェックリストの全項目クリアが、プロジェクト成功の絶対条件です。
**一つでも仕様違反があれば、学習結果に影響し、プロジェクト全体が失敗します。**