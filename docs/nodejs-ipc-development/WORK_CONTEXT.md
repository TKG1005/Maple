# Node.js IPCサーバー開発 - 作業コンテキスト

## 現在の状況 (更新日時: 2025-07-30 14:30)

### 🎯 作業段階
- **現在フェーズ**: Phase A - 基盤修正・テスト開始準備完了
- **作業ブランチ**: `feature/node-ipc-server-development`
- **完了済み**: 詳細開発計画の策定、ブランチ作成
- **次のタスク**: Phase A.1 - 現行システム問題の特定と修正

### 📊 現在の実装状況

#### ✅ 完成済みコンポーネント
- IPCサーバー基本構造: `pokemon-showdown/sim/ipc-battle-server.js` (614行)
- Python IPCBattle: `src/sim/ipc_battle.py` (244行)  
- IPCBattleFactory: `src/sim/ipc_battle_factory.py` (150行)
- デュアルモード通信システム: Phase 1-3完了済み

#### ⚠️ 既知の問題点
1. **種族名フォーマット不一致**: 
   - ダメージ計算器: `KeyError: 'ditto'`
   - IPCBattle生成: 小文字 "ditto"
   - species_mapper要求: 大文字 "Ditto"

2. **BattleStream統合不完全**:
   - 基本ping-pong通信は動作
   - 実際のバトル作成でタイムアウト発生
   - Pokemon Showdown依存関係の問題

3. **環境統合の不安定性**:
   - `env.reset()`でのタイムアウト
   - IPCプロセスの適切な初期化待機

### 🧪 最新のテスト結果
```bash
# 基本通信テスト
cd pokemon-showdown && echo '{"type":"ping"}' | node sim/ipc-battle-server.js
✅ 結果: {"type":"pong","success":true}

# フルIPC訓練テスト  
python train.py --full-ipc --battle-mode local --episodes 1 --parallel 1
⚠️ 結果: IPC通信確立後、env.reset()でタイムアウト

# 現在のエラー箇所
❌ ダメージ計算器: "ditto"種族名認識エラー
❌ 環境リセット: _create_ipc_battles()内でタイムアウト
```

### 📋 次の具体的作業 (Phase A.1 - 重要性大幅UP)

#### 🚨 **最重要**: Pokemon Showdown完全仕様準拠 + 不完全情報ゲーム対応
**プロジェクト成功の絶対条件**: 
1. ShowdownサーバーのAPIを100%正確に再現
2. **各EnvPlayerが独立したBattleオブジェクトを持つ**
3. **MapleShowdownCoreがプレイヤー固有メッセージを適切に振り分ける**

#### 1. BattleStream API正確実装 (最優先)
- [ ] `const { BattleStream } = require('./dist/sim/battle-stream')`の確認
- [ ] 正確な初期化: `new BattleStream({ debug: false, noCatch: false, keepAlive: true })`
- [ ] `ObjectReadWriteStream<string>`継承の理解

#### 2. Protocol Message完全準拠 (CRITICAL) 
- [ ] `>start {"formatid":"gen9randombattle"}`の正確な送信
- [ ] `>player p1 {"name":"Player1","team":null}`フォーマット確認
- [ ] `update\nMESSAGES` vs `sideupdate\nPLAYERID\nMESSAGES`の区別
- [ ] 🚨 **不完全情報ゲーム要件**: プレイヤー固有メッセージフィルタリング実装

#### 3. SIM-PROTOCOL.md完全理解
- [ ] 全Major/Minor actionsのメッセージ形式確認
- [ ] Pokemon ID format: `p1a: Pikachu`の正確な実装
- [ ] Choice request JSONフォーマット詳細確認

#### 4. Teams API統合
- [ ] `Teams.generate('gen9randombattle')`の正確な使用
- [ ] Packed format vs JSON format理解
- [ ] Species名正規化システム

### 🛠️ 次回作業開始時の手順

#### 1. 環境復元
```bash
cd /Users/takagikouichi/GitHub/Maple
git checkout feature/node-ipc-server-development
# 依存関係確認
cd pokemon-showdown && npm list | grep battle-stream
```

#### 2. 現状確認テスト
```bash
# 基本IPC通信確認
cd pokemon-showdown && timeout 5s node sim/ipc-battle-server.js <<< '{"type":"ping"}'

# Python統合テスト
python train.py --full-ipc --episodes 1 --parallel 1 --log-level DEBUG
```

#### 3. 作業開始
```bash
# A.1.1: BattleStream問題の特定
cd pokemon-showdown
node -e "
const { BattleStream } = require('./dist/sim/battle-stream');
console.log('BattleStream available:', !!BattleStream);
try { 
  const stream = new BattleStream(); 
  console.log('Initialization successful');
} catch(e) { 
  console.log('Error:', e.message); 
}
"
```

### 📈 パフォーマンス目標の追跡

#### 現在のベースライン (WebSocketモード)
- 通信遅延: 10-15ms
- 環境step処理時間: 11.7% (WebSocket通信部分)
- 1000ステップ実行時間: 15秒

#### IPC目標
- 通信遅延: 1-2ms (90%削減)
- WebSocket通信: 0% (完全排除)
- パフォーマンス向上: 75%以上

#### 測定方法
```bash
# ベースライン測定
python train.py --battle-mode online --profile --episodes 5

# IPC性能測定
python train.py --full-ipc --profile --episodes 5
```

### 🚨 注意事項・制約

#### 開発環境固有の問題
- **Node.js版本**: v16+ 必須
- **TypeScript→JavaScript**: dist/ディレクトリの依存
- **poke-env互換性**: CustomBattleクラスとの継承関係維持

#### メモリ・プロセス管理
- Node.jsプロセスの適切なクリーンアップ
- 長時間実行時のメモリリーク監視
- 並列実行時の競合状態回避

### 🔍 デバッグ・トラブルシューティング

#### よく使用するデバッグコマンド
```bash
# IPCプロセスの状態確認
ps aux | grep "node.*ipc-battle-server"

# メモリ使用量監視  
while true; do ps -o pid,vsz,rss,comm -p $(pgrep -f ipc-battle-server); sleep 2; done

# 詳細ログでの実行
python train.py --full-ipc --log-level DEBUG --episodes 1 2>&1 | tee debug.log
```

#### 緊急時のフォールバック
```python
# WebSocketモードでの動作確認
python train.py --battle-mode online --episodes 1

# 部分的なIPCテスト
python -c "
from src.sim.battle_communicator import IPCCommunicator
import asyncio
async def test(): 
    comm = IPCCommunicator()
    await comm.connect()
    response = await comm.send_message({'type':'ping'})
    print('Response:', response)
asyncio.run(test())
"
```

### 📝 作業ログ

#### 2025-07-30 14:30 - 開発計画策定完了
- 12-18日の詳細開発計画作成
- Phase A-D の段階的実装戦略確立
- コンテキスト保持システム構築
- 作業ブランチ `feature/node-ipc-server-development` 作成

#### 次回更新予定
- Phase A.1作業開始時
- 最初の技術的問題解決時  
- 各フェーズ完了時

---

**重要**: 作業中断時は必ずこのファイルを更新し、現在の進捗・問題・次の作業を記録してください。