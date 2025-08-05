<!--
  MapleShowdownCore リファクタリング計画
  Node.js IPC バトルサーバ (MapleShowdownCore) の冗長・非互換部分を整理し、
  純正の Showdown プロトコルに近づける設計案。
-->
# MapleShowdownCore リファクタリング計画

## 1. 背景と目的
- 現状の IPCBattleserver (`ipc-battle-server.js`) は `BattleStream.on('data')` を想定する古い API を利用
- JSON→テキスト→JSON の二重ラップ、不要な状態保存・復元、未対応イベントなどで複雑化
- 純正 Showdown テキストプロトコルに沿ったシンプルかつ透過的な実装を目指す

## 2. 主な問題点
1. **BattleStream API ミスマッチ**  
   `BattleStream` は EventEmitter ではなく AsyncIterable／`BattleTextStream` を使う必要がある
2. **バトル開始コマンドの非互換**  
   `>start ${JSON.stringify(...)}` の一行起動は非標準。純正の `|start|…`,`|player|…` 等に揃える
3. **冗長な JSON ラッピング**  
   Node.js 側で JSON 化・再パースせず、テキストをそのまま流し Python 側で必要整形が望ましい
4. **対応メッセージの不足・フィルタリング**  
   `|teamsize|`,`|teampreview|`,`|sideupdate|` など多くのプロトコル行が除外されている
5. **不要機能の混在**  
   `save_battle_state`/`restore_battle_state` 機能は自己対戦用途ではほぼ不要で複雑化要因
6. **ログ・エラーハンドリングの冗長**  
   stderr と stdout に重複ログ。Python 側は stdout の JSON のみ利用すべき
7. **リソース管理不足**  
   `this.battles` Map が永続的に肥大化し、`destroy_battle` でも stream 解放が不完全
8. **stdin readline の脆弱性**  
   チーム文字列など多行メッセージを安全に受信できない可能性あり

## 3. リファクタリング方針
- Node.js 側はエラーメッセージを stderr に、IPCプロトコルメッセージとShowdownメッセージを stdout に出力
- すべてのstdout出力はJSON形式に統一（IPCプロトコルメッセージとShowdownプロトコルメッセージ両方）
- IPCプロトコルメッセージには`type`フィールドで識別子を付与
- Showdownオリジナルメッセージは`{"type": "protocol", "data": "..."}`形式でラップして互換性を保持
- Python側のIPCCommunicatorはstdoutのみを監視し、JSONをパースして処理
- `save/restore` 周りは削除。エラーは JSON エラーオブジェクトのみ stdout へ出力
- active battles Map は単一バトル前提に簡素化。`destroy_battle` で必ず解放

## 4. 作業ステップ
1. **BattleStream→BattleTextStream/AsyncIterable 置換**  
   - `ipc-battle-server.js` の `new BattleStream()` 部分を `BattleTextStream` に変更
   - `.on('data')` → `battleTextStream.on('data')` あるいは `for await(...)` 形式で受信
2. **純正プロトコルによるバトル生成**  
   - JSON埋め込み起動から、テキストコマンド列 `|start|format`, `|player|…`, `|teamsize|…` へ移行
3. **JSON形式での統一出力**  
   - Node 側はすべてのメッセージをJSON形式でstdoutへ出力
   - Showdownプロトコルメッセージ：WebSocket形式と同じく複数行を改行で結合した1つの文字列
   - 形式：`{"type": "protocol", "battle_id": "...", "data": ">battle-format-id\n|init|battle\n..."}`
   - IPCプロトコルメッセージ：`{"type": "battle_created", "battle_id": "...", ...}`
   - Python 側の `IPCCommunicator` で統一的にJSONパースして処理
4. **不要機能の除去**  
   - `save_battle_state`/`restore_battle_state`/`list_saved_states` などを削除
5. **エラーハンドリングの統一**  
   - stderr 出力は最小限に留め、stdout でのみ JSON エラー出力
6. **バトル管理の簡素化**  
   - Map は単一バトルまたは必要最小限のインスタンス管理へ変更
   - `destroy_battle` で必ず BattleStream を `.destroy()` などで解放
7. **stdin 受信ロジックの強化**  
   - `readline` から `input.on('data')` + 自前バッファリングへ変更し、任意長行を安全に処理
8. **ドキュメント & テスト更新**  
   - `SHOWDOWN_MESSAGE_SPEC.md` に合わせプロトコル例を更新
   - Node.js サーバのユニットテスト or e2e テストを追加

## 5. テスト・検証 
- 単一バトル起動 → テキストプロトコル出力検証  
- Python + IPCCommunicator 経由で `create_battle` から `battle_end` まで正常フロー
- エラーケース (不正 format, missing params) の JSON エラー検証

## 6. 参考資料
- SIM-PROTOCOL.md (pokemon-showdown ソース)
- SHOWDOWN_MESSAGE_SPEC.md (Maple 側 IPC メッセージ仕様)
   
---
*この計画書をベースに、まず BattleStream 周りの API 修正から着手してください。*