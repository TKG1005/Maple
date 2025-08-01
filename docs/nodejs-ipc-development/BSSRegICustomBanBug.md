<!--
  Documentation: Bug report for Gen9 BSS Reg I custom ban check in IPC mode
-->
# Gen9 BSS Reg I におけるカスタムBANチェックの誤判定バグ

## 概要
- エラー: `STREAM_READ_ERROR: Custom bans are not currently supported in [Gen 9] BSS Reg I.`
- 発生環境: ローカル IPC モード (`--battle-mode local --full-ipc`)

## 再現手順
1. `train.py --battle-mode local --full-ipc --team random` を実行
2. Node.js IPC サーバー起動時に以下のログが出力される:
   ```
   🟡 Node.js stderr: Created battle: 1-xxxxxx (gen9bssregi)
   🟡 Node.js stderr: STREAM_READ_ERROR: Custom bans are not currently supported in [Gen 9] BSS Reg I.
   ```

## 原因分析
- **Showdown ランダムチーム生成モジュール** (`data/random-battles/gen9/teams.ts`) 内の
  `RandomTeams.hasDirectCustomBanlistChanges()` が、
  `format.restricted` の要素（公式制限）のみで「カスタムBAN変更あり」と誤判定。
- その後の `enforceNoDirectCustomBanlistChanges()` が例外を投げ、IPC クライアント層で `STREAM_READ_ERROR` に。

## WebSocket モードとの違い
- WebSocket モードではサーバー側チャットプラグイン
  (`dist/server/chat-plugins/randombattles`) を通し、
  別実装の乱数生成コードを使うため、上記チェックは走らずエラーが発生しない。

## 修正案
1. **ランダムチーム生成モジュールの修正**:
   ```ts
   // restricted===公式制限 は無視
   if (
     this.format.banlist.length
     || (this.format.restricted.length && this.format.customRules?.length)
     || this.format.unbanlist.length
   ) return true;
   ```
2. **Node.js 起動時モンキーパッチ**:
   環境変数 `NODE_OPTIONS="--require noCustomBanCheck.js"` で関数を no-op に差し替え
3. **Python 側でカスタムチームを明示的に渡す**:
   `DualModeEnvPlayer(..., team=<生チーム文字列>)` を使い、乱数生成ルートを回避

## 適用済みパッチ
- `teams.ts` / `dist/.../teams.js` に上記 (1) の変更を適用済み

---
*作成日: $(date '+%Y-%m-%d')*