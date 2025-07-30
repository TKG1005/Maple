# Node.js IPCサーバー開発 - 問題解決ログ

## 🚨 現在進行中の課題

### Issue #1: ダメージ計算器での種族名認識エラー (CRITICAL)
**発見日時**: 2025-07-30 14:00  
**優先度**: 🔴 高  
**状態**: 🔄 調査中

**問題詳細**:
```python
KeyError: 'ditto'
  File "src/damage/calculator.py", line 85, in calculate_damage
  File "src/utils/species_mapper.py", line 23, in get_species_info
```

**症状**:
- IPCBattleが小文字"ditto"でPokemonを作成
- ダメージ計算器が"ditto"を認識できない
- species_mapperは"Ditto"（大文字）を期待

**調査結果**:
- IPCBattle._create_minimal_teams()で`species='ditto'`を設定
- DamageCalculatorはspecies_mapperに依存
- species_mapperのキーは"Ditto", "Pikachu"等の大文字開始

**解決案**:
1. IPCBattleで種族名を"Ditto"に統一
2. species_mapperで小文字変換を追加
3. 正規化関数を作成して統一処理

**次のアクション**:
- [ ] species_mapper.pyの内容確認
- [ ] 正規化関数の実装場所決定
- [ ] IPCBattleの種族名生成ロジック修正

---

### Issue #2: 環境リセット時のタイムアウト (CRITICAL)
**発見日時**: 2025-07-30 14:15  
**優先度**: 🔴 高  
**状態**: 🔄 調査中

**問題詳細**:
```python
TimeoutError in PokemonEnv.reset()
  File "src/env/pokemon_env.py", line 245, in reset
  File "src/env/pokemon_env.py", line 298, in _create_ipc_battles
```

**症状**:
- `python train.py --full-ipc`実行時
- env.reset()で10秒タイムアウト
- IPC通信は確立されているが太多のバトル作成で停止

**調査結果**:
- IPCCommunicator.connect()は成功
- ping-pong通信は正常動作
- バトル作成リクエスト送信後に応答なし

**疑わしい原因**:
1. BattleStreamの初期化失敗
2. チーム情報のフォーマット問題
3. Node.jsプロセスでの例外発生

**次のアクション**:
- [ ] Node.js側のエラーログ確認
- [ ] BattleStream初期化の詳細調査
- [ ] バトル作成リクエストの内容検証

---

### Issue #3: BattleStream依存関係問題 (MEDIUM)
**発見日時**: 2025-07-30 14:30  
**優先度**: 🟡 中  
**状態**: 🔍 調査待ち

**問題詳細**:
```javascript
Error: Cannot find module '../dist/sim/battle-stream'
```

**症状**:
- ipc-battle-server.js実行時の不安定な動作
- 時々モジュールが見つからないエラー

**調査結果**:
- pokemon-showdown/dist/ディレクトリは存在
- battle-stream.jsファイルも存在
- パス解決の問題の可能性

**次のアクション**:
- [ ] dist/ディレクトリの内容確認
- [ ] require()パスの検証
- [ ] TypeScript→JavaScript変換の確認

---

## ✅ 解決済みの課題

### Issue #0: 基本IPC通信の確立 (RESOLVED)
**解決日時**: 2025-07-30 14:00  
**問題**: Node.jsプロセスとPython間のIPC通信
**解決策**: stdin/stdoutベースのJSON通信実装
**検証**: ping-pong通信で動作確認済み

---

## 📋 課題管理

### 優先度別整理
🔴 **Critical (即座対応必要)**:
- Issue #1: 種族名認識エラー
- Issue #2: 環境リセットタイムアウト

🟡 **Medium (Phase A中に対応)**:
- Issue #3: BattleStream依存関係

🟢 **Low (後のPhaseで対応)**:
- なし

### 依存関係
- Issue #2の解決にはIssue #3の調査が必要
- Issue #1は独立して解決可能
- Issue #3の解決でIssue #2が連鎖的に解決する可能性

### 解決予定スケジュール
- **今日 (7/30)**: Issue #1 + Issue #3調査
- **明日 (7/31)**: Issue #2解決、Phase A.1完了
- **8/1**: Phase A.2開始

---

## 🔧 デバッグ・検証方法

### Issue #1 検証方法
```python
# 種族名問題の確認
from src.sim.ipc_battle import IPCBattle
from src.damage.calculator import DamageCalculator

battle = IPCBattle('test', 'player', logger, communicator)
pokemon = list(battle._team.values())[0]
print(f"Pokemon species: '{pokemon.species}'")

calc = DamageCalculator()
try:
    calc.calculate_damage('tackle', pokemon.species, pokemon.species)
    print("✅ Damage calculation successful")
except KeyError as e:
    print(f"❌ KeyError: {e}")
```

### Issue #2 検証方法
```bash
# 詳細デバッグでの実行
python train.py --full-ipc --episodes 1 --parallel 1 --log-level DEBUG 2>&1 | tee issue2_debug.log

# Node.js側のエラー確認
cd pokemon-showdown
node sim/ipc-battle-server.js 2>&1 | tee nodejs_debug.log &
echo '{"type":"create_battle","battle_id":"test","format":"gen9randombattle","players":[{"name":"p1"},{"name":"p2"}]}' | nc localhost - || echo "Manual test"
```

### Issue #3 検証方法
```bash
# モジュール存在確認
ls -la pokemon-showdown/dist/sim/battle-stream*

# requireテスト
cd pokemon-showdown
node -e "
try {
  const { BattleStream } = require('./dist/sim/battle-stream');
  console.log('✅ Module loaded successfully');
  console.log('BattleStream:', typeof BattleStream);
} catch(e) {
  console.log('❌ Module load failed:', e.message);
}
"
```

---

## 📚 参考情報

### 関連ファイル
- `src/sim/ipc_battle.py`: IPCBattleクラス実装
- `src/damage/calculator.py`: ダメージ計算ロジック
- `src/utils/species_mapper.py`: Pokemon種族名マッピング
- `pokemon-showdown/sim/ipc-battle-server.js`: Node.js IPCサーバー
- `pokemon-showdown/dist/sim/battle-stream.js`: バトルエンジン

### 有用なコマンド
```bash
# プロセス監視
ps aux | grep -E "(node|python).*ipc"

# ログ監視
tail -f logs/train_*.log

# メモリ使用量
ps -o pid,vsz,rss,comm -p $(pgrep -f "ipc-battle-server")
```

---

**最終更新**: 2025-07-30 14:35  
**次回更新予定**: Issue解決時 or 新しい問題発見時