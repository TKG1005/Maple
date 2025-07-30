# Node.js IPC Battle Server 開発計画書

## 概要

WebSocket通信を完全に排除し、IPCによる高速バトル処理を実現するNode.js IPCサーバーの完全実装計画。Phase 4の最終段階として、既存のIPCBattleクラスと統合し、75%のパフォーマンス向上を達成する。

## 現状分析 (2025年7月30日時点)

### ✅ 完成済みコンポーネント
- **基本IPCサーバー構造**: `pokemon-showdown/sim/ipc-battle-server.js` (614行)
- **Python側IPCBattle**: `src/sim/ipc_battle.py` (244行)
- **IPCBattleFactory**: `src/sim/ipc_battle_factory.py` (150行)
- **デュアルモード通信**: `src/sim/battle_communicator.py`
- **環境統合**: PokemonEnv full-ipc対応済み

### 🔧 必要な修正・改善
- **BattleStream統合**: 現在の実装はプロトタイプレベル
- **チーム管理**: Pokemon種族情報の適切な処理
- **リアルタイム通信**: バトル進行状況の双方向同期
- **エラーハンドリング**: プロダクション環境での堅牢性

### 📊 目標性能
- **通信遅延**: 10-15ms → 1-2ms (90%削減)
- **WebSocket排除**: 11.7% → 0% (完全排除)
- **メモリ効率**: 既存+10%以内
- **安定性**: 100エピソード連続実行

## 段階的開発計画

### Phase A: Pokemon Showdown正確な仕様理解・統合 (4-6日)

#### A.1: BattleStream API完全準拠実装 (2-3日)
**目標**: Pokemon ShowdownのBattleStream仕様に100%準拠した実装

**重要な発見事項**:
- dist/sim/battle-stream.jsが存在し、正しいAPIが利用可能
- BattleStreamは`ObjectReadWriteStream<string>`を継承
- 正確な初期化とメッセージフォーマットが必要

**作業内容**:
1. **正確なBattleStream初期化**
   ```javascript
   const { BattleStream } = require('./dist/sim/battle-stream');
   const stream = new BattleStream({
     debug: false,
     noCatch: false, 
     keepAlive: true,
     replay: false
   });
   ```

2. **Pokemon Showdown Protocol完全準拠**
   ```javascript
   // 正しいバトル開始シーケンス
   stream.write(`>start {"formatid":"gen9randombattle"}`);
   stream.write(`>player p1 {"name":"${player1Name}","team":null}`); // random battle
   stream.write(`>player p2 {"name":"${player2Name}","team":null}`);
   ```

3. **メッセージハンドリングの正確な実装**
   - `update\nMESSAGES`形式の正確な解析
   - `sideupdate\nPLAYERID\nMESSAGES`の個別プレイヤー通信
   - `end\nLOGDATA`のバトル終了処理

#### A.2: Teams API統合とフォーマット対応 (1-2日)
**目標**: 正確なチーム生成・管理システムの実装

**作業内容**:
1. **Teams APIの正確な使用**
   ```javascript
   const { Teams } = require('./dist/sim/teams');
   
   // ランダムバトル用チーム生成
   const team = Teams.generate('gen9randombattle');
   const packedTeam = Teams.pack(team);
   
   // カスタムチーム処理
   const customTeam = Teams.importTeam(teamString);
   const validatedTeam = Teams.validate(customTeam);
   ```

2. **種族名・ID正規化システム**
   - Pokemon ShowdownのDex APIとの統合
   - species ID → display name変換
   - 大文字・小文字・ハイフン等の正規化

3. **チーム検証・互換性確保**
   - Team Validator APIの利用
   - 不正チーム検出・修正
   - フォーマット別制約の適用

**成果物**:
- 修正されたipc-battle-server.js
- 基本動作確認スクリプト
- 問題点記録ドキュメント

**検証方法**:
```bash
# 単体テスト
cd pokemon-showdown && node sim/ipc-battle-server.js
echo '{"type":"ping"}' | node sim/ipc-battle-server.js

# 統合テスト
python train.py --full-ipc --episodes 1 --parallel 1
```

#### A.2: Pokemon種族・チーム管理の改善 (1-2日)
**目標**: ダメージ計算器エラーを完全解決し、多様なPokemon対応

**作業内容**:
1. **種族名フォーマットの統一**
   - "ditto" vs "Ditto" vs "ditto-1" 問題の解決
   - species_mapper.pyとの統合
   - 英語名・ID・形態の正規化

2. **チーム生成の多様化**
   - 固定dittoチームから実用的なランダムチーム
   - config/teams/との統合
   - レベル50統計計算の正確性確保

3. **統計・能力値計算の最適化**
   - base_stats → stats変換の正確性
   - 努力値・個体値・性格補正の実装
   - タイプ効果計算の完全対応

**成果物**:
- 改善されたIPCBattle初期化
- Pokemon種族データベース統合
- チーム多様性テストスイート

**検証方法**:
```python
# 種族名テスト
battle = IPCBattle('test', 'player', logger, communicator)
print([p.species for p in battle._team.values()])

# ダメージ計算テスト
python -c "from src.damage.calculator import DamageCalculator; calc = DamageCalculator(); print(calc.calculate_damage('tackle', 'ditto', 'ditto'))"
```

#### A.3: エラーハンドリング・ログ改善 (1日)
**目標**: プロダクション環境での堅牢性確保

**作業内容**:
1. **包括的エラーキャッチ**
   - Node.js側のuncaughtException処理
   - Python側のIPCプロセス停止検出
   - タイムアウト・デッドロック回避

2. **詳細ログ・デバッグ情報**
   - バトル進行の詳細トレース
   - パフォーマンス測定ポイント
   - 問題特定のためのコンテキスト情報

3. **自動回復メカニズム**
   - プロセス再起動
   - 状態復元
   - グレースフル・シャットダウン

**成果物**:
- 堅牢なエラーハンドリング実装
- 包括的ログシステム
- 自動回復テストケース

### Phase B: 完全Battle Protocol実装 (5-7日)

#### B.1: Choice Request System完全実装 (2-3日)
**目標**: Pokemon Showdownの選択システムを100%再現

**重要な仕様確認**:
- Choice requestは`sideupdate\np1\nREQUEST`形式で送信
- REQUEST.activeに行動可能な技情報
- REQUEST.sideにチーム全体の状態情報
- RQID (request ID) による重複防止システム

**作業内容**:
1. **Request解析・管理システム**
   ```javascript
   // Choice requestの正確な処理
   parseChoiceRequest(requestData) {
     const request = JSON.parse(requestData);
     return {
       rqid: request.rqid,
       active: request.active,  // 行動可能技・対象
       side: request.side,      // チーム状態
       forceSwitch: request.forceSwitch || false
     };
   }
   ```

2. **Decision Validation System**
   ```javascript
   // 正確な行動検証
   validateChoice(choice, request) {
     // move 1, switch 2, team 123456等の検証
     // 対象指定の検証 (+1, -1等)
     // Mega evolution, Z-move, Dynamax制約
   }
   ```

3. **Turn Processing完全実装**
   - 両プレイヤーの行動収集待機
   - 優先度計算・同時処理
   - ターン結果の正確な配信

#### B.2: Protocol Message完全対応 (2-3日)
**目標**: SIM-PROTOCOL.mdの全メッセージタイプに対応

**作業内容**:
1. **Major Actions完全実装**
   ```javascript
   // |move|POKEMON|MOVE|TARGET
   // |switch|POKEMON|DETAILS|HP STATUS  
   // |drag|POKEMON|DETAILS|HP STATUS
   // |faint|POKEMON
   ```

2. **Minor Actions包括対応**
   ```javascript
   // |-damage|POKEMON|HP STATUS
   // |-heal|POKEMON|HP STATUS
   // |-status|POKEMON|STATUS
   // |-boost|POKEMON|STAT|AMOUNT
   // |-weather|WEATHER
   // |-fieldstart|CONDITION
   ```

3. **Battle Flow Management**
   - Team preview phase処理
   - Turn-by-turn progression
   - Battle end conditions (win/tie)

**成果物**:
- 完全動作するバトルエンジン
- 全行動タイプ対応
- バトル状態同期システム

**検証方法**:
```javascript
// Node.js側テスト
const battle = createBattle('test', 'gen9randombattle', players);
battle.stream.write('>p1 move 1');
const updates = battle.stream.read();
console.log('Battle updates:', updates);
```

#### B.2: リアルタイムバトル進行システム (1-2日)
**目標**: Pythonクライアントとの完全同期

**作業内容**:
1. **ターンベース進行管理**
   - 両プレイヤーの行動待ち
   - 同時実行の適切な処理
   - priority計算の反映

2. **状態更新イベント処理**
   - HP変化、状態異常、能力変化
   - 場の効果（天候、フィールド等）
   - Pokemon交代・倒れる処理

3. **バトル終了判定**
   - 勝敗の確定
   - 結果データの整理
   - リプレイデータの生成

**成果物**:
- リアルタイム進行システム
- イベントドリブン状態管理
- バトル終了処理

#### B.3: パフォーマンス最適化・測定 (1日)
**目標**: 目標75%削減の達成と検証

**作業内容**:
1. **通信レイテンシ最適化**
   - JSON serialization/deserializationの最適化
   - バッファリング戦略
   - 無駄な通信の排除

2. **メモリ使用量最適化**
   - オブジェクトプールの活用
   - 不要データの適切なクリーンアップ
   - GC圧力の最小化

3. **包括的パフォーマンステスト**
   - WebSocket vs IPC比較測定
   - 大量並列実行テスト
   - 長時間実行安定性テスト

**成果物**:
- 最適化されたIPCサーバー
- パフォーマンス比較レポート
- ベンチマーク自動化スクリプト

### Phase C: Python-NodeJS統合システム (4-5日)

#### C.1: IPCBattle完全リファクタリング (2-3日)
**目標**: Pokemon Showdown仕様に100%準拠したPython側実装

**重要な統合ポイント**:
- 現在のIPCBattleはダミーデータ（"ditto"）生成
- 実際のShowdownからの状態データ反映が必要
- poke-env互換性の完全維持

**作業内容**:
1. **Real Battle Data Integration**
   ```python
   class IPCBattle(CustomBattle):
       def __init__(self, battle_tag, username, logger, communicator):
           # Node.jsから実際のバトルデータを取得
           self._sync_battle_state_from_node()
           
       def _sync_battle_state_from_node(self):
           # BattleStreamからの状態同期
           # 実際のPokemon種族・技・統計値
           # HP, status, boosts等の完全反映
   ```

2. **Choice Action Integration** 
   ```python
   async def choose_move(self, battle):
       # Python側の行動 → Node.js送信
       # Choice requestの受信・解析
       # 行動結果の状態反映
   ```

3. **Live State Synchronization**
   - ターン進行の完全同期
   - ダメージ・回復の即座反映
   - 状態異常・ブーストの管理

#### C.2: Performance Integration Testing (1-2日)
**目標**: 本格的な訓練環境での性能検証

**成果物**:
- 完全なバトル状態管理システム
- 状態操作API
- 研究用ツール群

#### C.2: マルチバトル・並列処理対応 (1-2日)
**目標**: 本格的な訓練環境での安定動作

**作業内容**:
1. **複数バトル同時管理**
   - バトルID管理とリソース分離
   - メモリ効率的な多重化
   - 競合状態の回避

2. **プロセスプール管理**
   - 複数IPCサーバープロセス
   - 負荷分散・障害回復
   - プロセス間通信最適化

3. **大規模並列テスト**
   - 100+環境での安定性
   - メモリリーク検出
   - パフォーマンス劣化の監視

**成果物**:
- 本格的な多重化システム
- 並列処理管理ツール
- 大規模テストスイート

### Phase D: プロダクション準備・最終統合 (2-3日)

#### D.1: 最終統合テスト・バグ修正 (1-2日)
**目標**: 全機能の統合動作確認

**作業内容**:
1. **エンドツーエンドテスト**
   - train.py完全実行テスト
   - 全学習アルゴリズムでの動作確認
   - エラーケースの網羅的テスト

2. **互換性テスト**
   - 既存WebSocketモードとの結果比較
   - 学習結果の一貫性確認
   - poke-env互換性の確認

3. **最終バグ修正**
   - 残存する不具合の解決
   - エッジケースの処理
   - 性能問題の最終調整

**成果物**:
- 完全動作するシステム
- 包括的テストレポート
- バグ修正履歴

#### D.2: ドキュメント・運用準備 (1日)
**目標**: 運用・保守のための情報整備

**作業内容**:
1. **運用ドキュメント**
   - セットアップ・設定手順
   - トラブルシューティングガイド
   - パフォーマンス調整方法

2. **開発者ドキュメント**
   - アーキテクチャ詳細説明
   - 拡張・カスタマイズ方法
   - APIリファレンス

3. **品質保証**
   - コードレビュー・リファクタリング
   - セキュリティチェック
   - 最終性能測定

**成果物**:
- 完全なドキュメント群
- 運用ツール・スクリプト
- 品質保証レポート

## コンテキスト保持・作業再開戦略

### 各段階での記録すべき情報

#### 作業開始時の記録 (WORK_CONTEXT.md)
```markdown
## 現在のコンテキスト (更新日時: YYYY-MM-DD HH:MM)

### 作業段階
- 現在フェーズ: Phase A.2
- 完了済み: A.1 (BattleStream初期化修正)
- 次のタスク: Pokemon種族管理の改善

### 現在の問題点
- ダメージ計算器での"ditto" vs "Ditto"エラー
- species_mapper.pyとの統合不完全

### 最新のテスト結果
- ping-pong通信: ✅ 正常
- バトル作成: ⚠️ 種族名エラー
- フルIPC実行: ❌ timeout

### 次の具体的作業
1. src/damage/calculator.pyの種族名正規化
2. IPCBattle._create_minimal_teams()の改善
3. species_mapper統合テスト
```

#### 問題・解決策の記録 (ISSUES_LOG.md)
```markdown
## 問題解決ログ

### Issue #1: BattleStream初期化失敗 (解決済み)
**問題**: `new BattleStream()`でmodule not found
**原因**: dist/sim/battle-streamのパス問題
**解決**: require('../dist/sim/battle-stream')に修正
**日時**: 2025-07-30 14:00

### Issue #2: 種族名フォーマット不一致 (進行中)
**問題**: DamageCalculatorが"ditto"を認識しない
**症状**: KeyError: 'ditto' in damage calculation
**調査**: species_mapperでは"Ditto"が正しい形式
**案**: IPCBattleで種族名を統一変換
**日時**: 2025-07-30 14:30
```

#### 進捗追跡 (PROGRESS.md)
```markdown
## 開発進捗

### Phase A: 基盤修正・テスト (3-5日)
- [x] A.1.1: BattleStream初期化修正
- [x] A.1.2: 基本通信テスト
- [ ] A.2.1: 種族名フォーマット統一 (進行中 50%)
- [ ] A.2.2: チーム生成多様化
- [ ] A.2.3: 統計計算最適化
```

### ファイル組織・バックアップ戦略

#### 重要ファイルのバックアップ
```bash
# 作業開始時に現状をバックアップ
mkdir -p backup/$(date +%Y%m%d_%H%M%S)
cp pokemon-showdown/sim/ipc-battle-server.js backup/$(date +%Y%m%d_%H%M%S)/
cp src/sim/ipc_battle.py backup/$(date +%Y%m%d_%H%M%S)/
cp src/damage/calculator.py backup/$(date +%Y%m%d_%H%M%S)/
```

#### 段階的コミット戦略
```bash
# 各段階完了時に詳細コミット
git add -A
git commit -m "feat(phase-a1): Fix BattleStream initialization

- Resolve module import path issues
- Add proper error handling for stream creation
- Implement basic ping-pong communication test
- Fix timeout issues in IPC communication

Phase A.1 完了: 基本通信確立
次の作業: A.2 Pokemon種族管理改善

Testing:
  ✅ ping-pong communication
  ✅ basic IPC connection
  ⏳ battle creation (种族名问题残存)
"
```

### 再開時のクイックスタート手順

#### 1. 現状確認スクリプト
```bash
#!/bin/bash
# quick-status.sh
echo "=== Current Status ==="
echo "Branch: $(git branch --show-current)"
echo "Last commit: $(git log -1 --oneline)"
echo "Modified files: $(git status --porcelain | wc -l)"
echo ""
echo "=== Latest context ==="
cat docs/WORK_CONTEXT.md | head -20
echo ""
echo "=== Quick test ==="
timeout 10s python train.py --full-ipc --episodes 1 --parallel 1 2>&1 | head -10
```

#### 2. 環境復元スクリプト
```bash
#!/bin/bash
# restore-env.sh
cd /Users/takagikouichi/GitHub/Maple
git checkout feature/node-ipc-server-development
source venv/bin/activate  # if using venv
pip install -r requirements.txt
cd pokemon-showdown && npm install
echo "Environment restored. Run ./quick-status.sh for current state."
```

### 品質保証・テスト戦略

#### 継続的テストスイート
```python
# test_ipc_development.py
def test_basic_communication():
    """Phase A完了の検証"""
    assert ping_pong_test()
    assert basic_battle_creation()

def test_pokemon_species():
    """Phase A.2完了の検証"""
    assert species_name_consistency()
    assert damage_calculator_integration()

def test_battle_engine():
    """Phase B完了の検証"""
    assert full_battle_execution()
    assert battle_state_synchronization()
```

#### パフォーマンス監視
```bash
# performance-check.sh
echo "=== Performance baseline ==="
python train.py --profile --episodes 5 --parallel 5 | grep "env_step"
echo ""
echo "=== IPC performance ==="
python train.py --full-ipc --profile --episodes 5 --parallel 1 | grep "env_step"
```

## Pokemon Showdown準拠検証基準

### 🎯 完全準拠基準 (最重要)
1. **Protocol Message互換性**:
   - [ ] SIM-PROTOCOL.mdの全メッセージタイプ対応
   - [ ] Choice request format完全一致  
   - [ ] Battle initialization sequence準拠
   - [ ] Team preview/turn progression正確実装

2. **Battle Mechanics完全再現**:
   - [ ] ダメージ計算100%一致（既存WebSocketとの比較）
   - [ ] 状態異常・能力変化の完全同期
   - [ ] 技効果・特性発動の正確な再現
   - [ ] 優先度・順序決定ロジック一致

3. **Teams & Format互換性**:
   - [ ] gen9randombattle完全対応
   - [ ] Packed format teams正確処理
   - [ ] Species/Move ID正規化100%
   - [ ] Team validation準拠

### 🚀 性能・統合基準
1. **パフォーマンス目標**:
   - [ ] WebSocket通信完全排除（11.7% → 0%）
   - [ ] IPC通信遅延 <2ms
   - [ ] `python train.py --full-ipc --episodes 100 --parallel 20` 完全実行
   - [ ] 既存学習結果との一貫性 (±3%以内)

2. **安定性要件**:
   - [ ] 1000エピソード連続実行
   - [ ] メモリリーク 0件（24時間）
   - [ ] 並列25環境での安定動作
   - [ ] エラー回復メカニズム完備

### 段階別完了判定
- **Phase A完了**: 基本IPCバトル1エピソード成功
- **Phase B完了**: リアルタイムバトル10エピソード成功  
- **Phase C完了**: 並列10環境で100エピソード成功
- **Phase D完了**: 全成功基準クリア

## リスク管理・緊急対応

### 高リスク項目
1. **Pokemon Showdown依存関係**: TypeScript/JavaScript互換性問題
2. **メモリ管理**: Node.jsプロセスリーク
3. **競合状態**: 並列実行時のデータ競合

### 緊急対応手順
1. **作業中断時**: 現状をWORK_CONTEXT.mdに記録、現在の変更をコミット
2. **問題発生時**: ISSUES_LOG.mdに詳細記録、バックアップファイルから復元
3. **期限遅延時**: Phase優先度見直し、MVP機能に集中

---

## 📋 作業チェックリスト

### 事前準備
- [ ] 作業ブランチ作成 (`feature/node-ipc-server-development`)
- [ ] バックアップディレクトリ作成
- [ ] WORK_CONTEXT.md初期化
- [ ] 基本動作テスト実行

### Phase A: 基盤修正・テスト
- [ ] A.1: BattleStream統合修正
- [ ] A.2: Pokemon種族管理改善  
- [ ] A.3: エラーハンドリング強化

### Phase B: バトルエンジン統合
- [ ] B.1: BattleStream完全統合
- [ ] B.2: リアルタイム進行システム
- [ ] B.3: パフォーマンス最適化

### Phase C: 高度機能
- [ ] C.1: 状態保存・復元システム
- [ ] C.2: 並列処理対応

### Phase D: 最終統合
- [ ] D.1: 統合テスト・バグ修正
- [ ] D.2: ドキュメント・運用準備

### 最終確認
- [ ] 全成功基準クリア
- [ ] パフォーマンス目標達成
- [ ] ドキュメント完成
- [ ] 運用準備完了

---

**総推定工数**: 12-18日 (Phase A: 3-5日, B: 4-6日, C: 3-4日, D: 2-3日)  
**開始日**: 2025年7月30日  
**目標完了日**: 2025年8月15日  
**作業ブランチ**: `feature/node-ipc-server-development`

このプランに従って段階的に実装を進め、各フェーズで確実に動作確認しながら最終的なWebSocket完全排除システムを完成させます。