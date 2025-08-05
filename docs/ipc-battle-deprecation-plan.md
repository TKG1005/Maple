# IPCBattle廃止計画

## 概要

IPCBattleクラスを廃止し、IPCClientWrapperに統合してアーキテクチャの重複を解消する。

## 背景・目的

### 変更前の問題
- **機能重複**: IPCBattleとIPCClientWrapperで戦闘コマンド送信・状態取得が重複
- **複雑な統合**: showdown ↔ IPC ↔ IPCBattle ↔ poke-env の多層構造
- **保守性問題**: 変更時に複数クラスの更新が必要

### 変更後の構造
```
【Before】showdown ↔ IPC ↔ IPCBattle ↔ poke-env (複雑)
                       ↕
                IPCClientWrapper (重複)

【After】 showdown ↔ IPC ↔ IPCClientWrapper ↔ poke-env (統合)
                            ↓
                    PSClient互換インターフェース
```

## 実装結果

### ✅ Phase 1: IPCClientWrapper拡張 (完了)
**コミット**: `3320a426c`

- **PSClient互換性**: AccountConfiguration/ServerConfiguration対応
- **認証システム**: `log_in()`、`wait_for_login()`実装
- **メッセージ処理**: showdownプロトコルとIPC制御メッセージの自動判別
- **poke-env統合**: `_handle_message()`による直接統合

### ✅ Phase 2: DualModePlayer統合 (完了)
**コミット**: `bd1548ec5`

- **初期化統合**: `_initialize_communicator()`でIPCClientWrapper作成
- **WebSocket置換**: `ps_client`をIPCClientWrapperで完全置換
- **接続フロー**: PSClient互換の`listen()`による接続確立

### ✅ Phase 3: IPCBattle削除 (完了)
**コミット**: `e7b0c6b9c`

- **ファイル削除**: `ipc_battle.py`、`ipc_battle_factory.py`を完全削除
- **参照除去**: 全インポート文・メソッド呼び出しを削除
- **エラー修正**: pokemon_env.pyのIndentationError解消

## 技術的成果

### アーキテクチャ簡素化
- **コード削減**: 1,004行の重複コード削除
- **責任明確化**: IPCClientWrapperが唯一のIPC通信責任点
- **統合インターフェース**: PSClient互換によりpoke-env統合簡素化

### 互換性維持
- **外部API**: DualModeEnvPlayerのインターフェース変更なし
- **設定フロー**: AccountConfiguration/ServerConfiguration継続使用
- **コマンドライン**: `--battle-mode online/local`動作継続

### 削除対象・修正ファイル
```
【削除】
src/sim/ipc_battle.py (494行)
src/sim/ipc_battle_factory.py (319行)
simple_teampreview_debug.py (191行)

【修正】
src/env/dual_mode_player.py: IPCClientWrapper拡張 (+242行)
src/env/pokemon_env.py: IPCBattleFactory呼び出し削除 (-67行)
```

## 残作業

### 📋 Phase 4: テスト・ドキュメント更新 (未実施)
- [ ] IPCClientWrapper拡張機能のテストコード作成
- [ ] オンライン・ローカルモード統合テスト
- [ ] CLAUDE.md等ドキュメント更新

---

**作成**: 2025-01-05 | **更新**: 2025-01-05 | **ステータス**: Phase 3完了