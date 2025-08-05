# Node.js IPC Development Documentation

## 📂 ドキュメント構成

このフォルダには、Pokemon Showdown IPCサーバー開発に関連する全ドキュメントが含まれています。

### 🎯 **最重要ファイル（必読順）**

#### 1. **現在の作業状況** 
- **`WORK_CONTEXT.md`** - 現在の作業段階・問題・次のタスク
- **`PROGRESS.md`** - Phase A-D の段階別進捗追跡  
- **`ISSUES_LOG.md`** - 技術的問題管理・解決策記録

#### 2. **開発計画・仕様** 
- **`node-ipc-server-development-plan.md`** - 12-18日の詳細開発計画
- **`SHOWDOWN_SPEC_COMPLIANCE.md`** - 🚨 **最重要** Pokemon Showdown仕様準拠チェックリスト

#### 3. **アーキテクチャ文書**
- **`showdown-integration-plan.md`** - Phase 1-4の統合実装計画全体  
- **`ipc-battle-architecture.md`** - IPC通信アーキテクチャ仕様

## 🚀 **コンテキスト復元手順**

コンテキストが喪失した場合、以下の順序で読み込み：

### **Step 1: 現状把握**
```bash
# 現在の作業状況を確認
cat WORK_CONTEXT.md | head -50
cat PROGRESS.md | grep -A 10 "現在の作業"
```

### **Step 2: 重要課題確認**  
```bash
# 現在の技術的問題を確認
cat ISSUES_LOG.md | grep -A 20 "現在進行中の課題"
```

### **Step 3: 仕様要件確認**
```bash
# Pokemon Showdown準拠要件を確認
cat SHOWDOWN_SPEC_COMPLIANCE.md | head -100
```

## 🎯 **プロジェクト成功の核心**

**最重要**: Pokemon Showdownサーバーの**100%完璧な再現**

- BattleStream API完全準拠
- SIM-PROTOCOL.md全メッセージ対応  
- チームデータ JSON 受け渡し（Node.js IPC サーバーでは Python から渡されたチーム情報をそのまま使用）
 - 生のShowdownプロトコル行（`>battle-…` や `|request|…` を含む各行）を一切変更せずそのまま送信し、Python側のpoke-env標準処理に完全委任
 - Node.jsプロセスは**raw protocol line**および制御用JSONメッセージ（`battle_created`/`battle_update`など）を**stdout**に書き出し、Python側へ一切加工せずに伝搬させる。デバッグログは**stderr**に出力する。
  
- Node.jsプロセスはオリジナルのPokemon Showdownサーバー動作をトレースし、同一のメッセージフォーマットとシーケンスを完全に再現して送受信する
- IPCサーバーは既存のWebSocket通信機能を最小限に置き換えることを目的とし、可能な限りShowdown準拠のNode.jsプロセスおよびpoke-env（EnvPlayer相当）の仕様をそのまま活用する

## 📋 **開発段階と対応ドキュメント**

### **Phase A: Pokemon Showdown仕様理解・統合** (4-6日)
- 主要参照: `SHOWDOWN_SPEC_COMPLIANCE.md`
- 実装ガイド: `node-ipc-server-development-plan.md` (Phase A章)

### **Phase B: 完全Battle Protocol実装** (5-7日)  
- 主要参照: `showdown-integration-plan.md`
- アーキテクチャ: `ipc-battle-architecture.md`

### **Phase C: Python-NodeJS統合システム** (4-5日)
- 統合計画: `node-ipc-server-development-plan.md` (Phase C章)
- 進捗追跡: `PROGRESS.md`

## ⚠️ **緊急時の最小参照**

完全にコンテキストが失われた場合、最低限以下を読み込み：

1. **`WORK_CONTEXT.md`** - 現在地点の確認
2. **`SHOWDOWN_SPEC_COMPLIANCE.md`** - 成功基準の確認  
3. **`ISSUES_LOG.md`** - 現在の問題確認

## 🔗 **関連外部ファイル**

### **実装ファイル**
```
/pokemon-showdown/sim/ipc-battle-server.js
/src/env/dual_mode_player.py (IPCClientWrapper)
/src/sim/battle_communicator.py
/docs/ipc-battle-deprecation-plan.md
```

### **Pokemon Showdown仕様**
```
/pokemon-showdown/PROTOCOL.md
/pokemon-showdown/sim/SIM-PROTOCOL.md  
/pokemon-showdown/sim/SIMULATOR.md
/pokemon-showdown/sim/TEAMS.md
```

---

**更新日**: 2025-01-05  
**作業ブランチ**: `feature/node-ipc-server-development`  
**現在の段階**: IPCClientWrapper統合完了、Node.jsサーバー開発中