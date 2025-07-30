# Node.js IPCサーバー開発 - コンテキスト復元ガイド

## 🚨 緊急時のClaude指示テンプレート

コンテキストが完全に失われた場合、以下をClaudeにそのまま送信してください：

---

## **Context Restore Request for Node.js IPC Server Development**

以下のファイルを読み込んで、Pokemon Showdown Node.js IPCサーバー開発のコンテキストを完全復元してください：

### **🎯 最優先参照ファイル（必読）**

#### 1. 現在の作業状況
```
/Users/takagikouichi/GitHub/Maple/docs/nodejs-ipc-development/WORK_CONTEXT.md
/Users/takagikouichi/GitHub/Maple/docs/nodejs-ipc-development/PROGRESS.md
/Users/takagikouichi/GitHub/Maple/docs/nodejs-ipc-development/ISSUES_LOG.md
```

#### 2. 開発計画・仕様要件
```
/Users/takagikouichi/GitHub/Maple/docs/nodejs-ipc-development/SHOWDOWN_SPEC_COMPLIANCE.md
/Users/takagikouichi/GitHub/Maple/docs/nodejs-ipc-development/node-ipc-server-development-plan.md
```

#### 3. 現在の実装状況
```
/Users/takagikouichi/GitHub/Maple/pokemon-showdown/sim/ipc-battle-server.js
/Users/takagikouichi/GitHub/Maple/src/sim/ipc_battle.py
/Users/takagikouichi/GitHub/Maple/src/sim/ipc_battle_factory.py
```

### **📋 Pokemon Showdown仕様参照**
```
/Users/takagikouichi/GitHub/Maple/pokemon-showdown/sim/SIM-PROTOCOL.md
/Users/takagikouichi/GitHub/Maple/pokemon-showdown/sim/SIMULATOR.md
/Users/takagikouichi/GitHub/Maple/pokemon-showdown/sim/TEAMS.md
```

### **🔗 プロジェクト全体理解**
```
/Users/takagikouichi/GitHub/Maple/CLAUDE.md
/Users/takagikouichi/GitHub/Maple/docs/nodejs-ipc-development/showdown-integration-plan.md
```

---

### **⚠️ 重要事項**

1. **プロジェクトの成功条件**: Pokemon Showdownサーバーの**100%完璧な再現**
2. **現在のブランチ**: `feature/node-ipc-server-development`
3. **総推定工数**: 13-18日（Phase A: 4-6日, Phase B: 5-7日, Phase C: 4-5日）
4. **核心技術要件**: 
   - BattleStream API完全準拠
   - SIM-PROTOCOL.md全メッセージ対応
   - Teams API正確統合
   - WebSocket通信100%排除

### **❓ 質問**

上記ファイルを読み込み後、以下を回答してください：

1. **現在の作業段階**は何ですか？（Phase A.X等）
2. **現在の主要な技術的問題**は何ですか？
3. **次に行うべき具体的なタスク**は何ですか？
4. **Phase X完了の判定基準**は何ですか？

---

## 🔄 **段階別コンテキスト復元**

### **Phase A作業中の場合**
追加で参照が必要なファイル：
```
/Users/takagikouichi/GitHub/Maple/pokemon-showdown/dist/sim/battle-stream.js
/Users/takagikouichi/GitHub/Maple/pokemon-showdown/sim/examples/battle-stream-example.ts
```

### **Phase B作業中の場合**  
追加で参照が必要なファイル：
```
/Users/takagikouichi/GitHub/Maple/src/sim/battle_communicator.py
/Users/takagikouichi/GitHub/Maple/src/env/dual_mode_player.py
```

### **Phase C作業中の場合**
追加で参照が必要なファイル：
```
/Users/takagikouichi/GitHub/Maple/src/env/pokemon_env.py
/Users/takagikouichi/GitHub/Maple/docs/nodejs-ipc-development/ipc-battle-architecture.md
```

## 📝 **作業再開時のクイック確認**

```bash
# 現在のブランチ確認
git branch --show-current

# 最新の変更確認  
git status --porcelain

# 最後のコミット確認
git log -1 --oneline

# 現在の作業内容確認
head -30 docs/nodejs-ipc-development/WORK_CONTEXT.md
```

## 🎯 **成功基準の確認**

開発再開前に必ず確認：
- [ ] Pokemon Showdown仕様100%準拠
- [ ] BattleStream API正確実装
- [ ] WebSocket vs IPC完全同一結果
- [ ] 1000エピソード安定実行

---

**最終更新**: 2025-07-30  
**使用方法**: このテンプレートをClaudeにコピペして送信