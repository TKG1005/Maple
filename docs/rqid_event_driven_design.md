# rQID Event-Driven Synchronization Design

本設計書は、rQID 待ちの 0.05 秒ポーリングを撤廃し、更新イベントで即時起床するイベント駆動方式へ移行するための詳細な設計・手順・検証計画を示す。作業が中断しても任意のセクションから再開可能なように細分化している。

---

## 1. Purpose & Scope
- 目的: rQID 同期のレイテンシと CPU アイドル消費を削減し、スループットを向上させる。
- 対象ファイル:
  - `src/env/pokemon_env.py`
    - rQID 同期ループ（約 1080–1112 行付近）
    - `race_get` のイベント競合時の微待機（約 723–726 行付近）
  - `src/env/dual_mode_player.py`
    - バトル準備待ち（約 713–727 行付近）
- 非対象: 学習アルゴリズム・報酬関数・PS サーバ側の挙動そのもの。

---

## 2. Current State Summary
- 複数箇所で「状態更新を待つための短周期 `sleep(0.05)`」が存在。
- rQID 同期では、`last_request.rqid` の更新を前提に短周期で再評価。
- `race_get` では、イベント先行時の瞬間的な空読みを避けるための微待機が存在。

---

## 3. Target Behavior
- rQID の更新を「発火条件」とし、更新イベント到達時に待機タスクを即時起床。
- ポーリングを原則撤廃。待機は「条件未達→待機登録→通知→起床→再検証」の枠組みに統一。
- rQID 以外の条件（`teamPreview`, `wait`, `forceSwitch` 等）には依存しない。

---

## 4. Constraints & Decisions
- Trigger: 「rqid が更新＝発火」の仕様で統一する。
- Failure: 失敗時はフォールバックせずエラー停止（明示的な例外を送出）。
- Observability First: メトリクス/ログを先に実装して、移行前後の効果を可視化。
- Cross-loop Safety: ループ跨ぎ（`POKE_LOOP` と他スレッド）を厳密に扱う。

---

## 5. Architecture Overview
- スレッド/ループ構成:
  - `POKE_LOOP`（PS クライアント/IPC 処理）と、学習/環境側の非同期処理が併存。
- 新規コンポーネント: RqidNotifier（バージョニング付き通知）
  - 役割: プレイヤー/バトル単位で最新の `rqid` と通知機構を保持。
  - データモデル: `player_id -> { version(int), last_rqid(int|None), state_meta(optional) }`
  - 同期モデル:
    - 案A: `POKE_LOOP` 内に `asyncio` ベースで設置し、他スレッドからは `run_coroutine_threadsafe` で publish/observe を呼ぶ。
    - 案B: スレッド安全な `threading.Condition`/`threading.Event` を用いた橋渡し。publish 側は version++ と通知、observe 側は baseline と比較しつつ wait。
  - 本設計では案Aを第一候補（イベントを `POKE_LOOP` に集約し、データ更新点と同一ループで整合性を担保）。

---

## 6. Public Interfaces (Conceptual)
- `publish_rqid_update(player_id, rqid, meta)`
  - いつ: `battle.last_request` 更新後（rqid 変化を検知した直後）。
  - 何を: 内部 `version` をインクリメント、`last_rqid` を更新、待機者へ通知。
- `wait_for_rqid_change(player_id, baseline_rqid, timeout)`
  - いつ: rQID の更新待ちをしたいとき。
  - 仕様: `last_rqid != baseline_rqid` になったら即時返す。`timeout` 超過でエラー停止。
- `register_battle(player_id, initial_rqid)` / `close_battle(player_id)`
  - 登録と解放。終了時は未処理待機者をエラーで起床させる。
- `emit_metric(event_name, fields)` / `log_event(tag, fields)`
  - メトリクス/ログの発火ポイントを統一。

---

## 7. Event Sources (Insertion Points)
- IPC 受信ポンプ（`dual_mode_player` 内の `_ipc_receive_pump`）
  - `battle.last_request` を in-place 更新する直後に `publish_rqid_update` を呼ぶ。
- バトル開始/終了ハンドリング
  - 開始時: `register_battle(player_id, initial_rqid)`
  - 終了時: `close_battle(player_id)`（待機中のタスクをエラーで起床）

---

## 8. Integration by File
- `src/env/dual_mode_player.py`
  - `_wait_for_battle_ready(battle_id)`
    - 現状: `last_request` が入るまで `sleep(0.05)` ポーリング。
    - 変更: 「初回 `last_request` 到着」を `publish_rqid_update` による通知で待機。待機前に現状態を確認（既に到着済みなら即時完了）。
- `src/env/pokemon_env.py`
  - `race_get(...)`
    - 現状: 競合解消でイベント後に最大 10 回の 0.05s 微待機。
    - 変更: キュー `put` と「finished_event」発火をイベント連鎖にし、`queue_updated` 通知で即リトライ。微待機は撤廃。
  - rQID 同期（約 1080–1112 行付近）
    - 現状: `pending` 集合に対して短周期 sleep を挟み再評価。
    - 変更: 各 `pid` ごとに `baseline_rqid = prev_rqids[pid]` を渡して `wait_for_rqid_change` を登録。`FIRST_COMPLETED` 的に待機→起床後に実条件を再評価し `pending` から除外。

---

## 9. Failure Handling Policy (No Fallback)
- タイムアウト: `wait_for_rqid_change(..., timeout)` 超過時に例外送出（従来の詳細ログ込み）。
- イベント未発火/取りこぼし: 常に「待機前に現状態を確認」し、`baseline` を明示。`publish` 側は必ず version++ を伴う。これでも条件未達ならタイムアウトで停止。
- 終了時: `close_battle` で待機者を例外起床し、リークを防止。

---

## 10. Metrics & Logging (Implement First)
- 優先実装（既存ポーリング版に先付け）
  - Metrics（名称/単位/概要）
    - `rqid_wait_latency_ms`: rQID 待機（チェック〜条件達成）に要した時間。
    - `rqid_poll_iterations`: rQID 同期のポーリング反復回数。
    - `rqid_timeouts_count`: rQID 待機のタイムアウト件数。
    - `event_publish_to_wakeup_ms`: publish から待機タスク起床までのレイテンシ。
    - `queue_put_to_get_latency_ms`: `race_get` のキュー put から get までの時間。
    - `micro_sleep_hits_count`: 微待機が実際に効いたケース数（現状把握用）。
    - `busy_wakeups_count`: 条件未達のまま起きたスパurious 起床数（イベント化後の健全性指標）。
    - `event_missing_error_count`: 想定イベント未到達で停止した件数。
  - Logs（構造化キー例）
    - `[RQSYNC] pid=... rqid_prev=... rqid_curr=... turn=... type=... dt_ms=...`
    - `[RQPUB] pid=... rqid_new=... version=... ts=...`
    - `[RQWAIT] pid=... baseline=... timeout=... result=ok|timeout dt_ms=...`
    - `[RACE] queue_put_to_get_ms=... finished_event=bool qsize_before=...`
  - 出力レベル: メトリクスは INFO/DEBUG 選択式。高頻度系はサンプリング比率を設定可能に（既定は 1.0）。
  - 検証方法: 移行前に 50〜100 試行で分布を採取し、移行後に同条件で比較。

---

## 11. Step-by-Step Plan (Fine-Grained, Resumable)
- S0: 計測基盤の最小追加
  - メトリクス/ログ API のダミー実装を用意し、既存ポーリング箇所にフック。
  - 完了判定: メトリクスがファイル/標準出力に出力される。`rqid_wait_latency_ms` が収集可能。
- S1: Baseline 収集
  - 現行（ポーリング）で N 試行の分布を取得。レポートを簡易集計。
- S2: RqidNotifier の骨組み（非公開/未使用）
  - publish/observe/close のインタフェースを用意（未配線）。
  - 完了判定: 単体で version 増加と待機解除がログで確認できる。
- S3: Publish 挿入
  - IPC 受信ポンプの `last_request` 更新箇所に `publish_rqid_update` を差し込む。
  - 完了判定: `[RQPUB]` ログが対戦で観測できる。
- S4: `_wait_for_battle_ready` をイベント待機に置換
  - 既存チェック→待機→再検証の順序を満たす形で observe を適用。
  - 完了判定: 0.05s ポーリングがなくなり、初回到着で即起床。
- S5: `race_get` 微待機の撤廃
  - `queue_updated` 通知を導入し、put/finished_event 連鎖で即再取得。
  - 完了判定: `micro_sleep_hits_count` が 0 に近づく。レイテンシ短縮を確認。
- S6: rQID 同期ループのイベント化
  - `pending` を baseline 付き observe に置換。`FIRST_COMPLETED` 的に進める。
  - 完了判定: `rqid_poll_iterations` が 0。イベント起床で処理が進む。
- S7: 失敗時のエラー停止整備
  - すべての待機で timeout→例外。詳細ログと共に停止。
- S8: Post-metrics 収集
  - 同条件で分布を再取得し、Before/After を比較。
- S9: 文書更新/除去判断
  - 残すメトリクス/ログの選別、閾値/サンプリング率の確定。

---

## 12. Verification & Acceptance Criteria
- 機能:
  - rQID 更新で待機が即時に解除される。
  - バトル準備・`race_get`・rQID 同期の全てでポーリング撤廃。
- 信頼性:
  - タイムアウト時に明確な例外/ログが出る。リークなし。
- 性能:
  - `rqid_wait_latency_ms` の中央値・95% が減少。
  - `rqid_poll_iterations` は 0、CPU アイドル時間が減少。

---

## 13. Risks & Mitigations
- レースコンディション（取りこぼし）
  - 待機前の現状態確認＋`baseline` 明示＋`publish` 時の version++ で低減。
- 同時発火スパイク
  - 起床後に対象再評価（差分ありのみ続行）。
- ループ跨ぎ不整合
  - Notifier を `POKE_LOOP` に集約。外部は `run_coroutine_threadsafe` 経由に限定。

---

## 14. Open Decisions
- Notifier 配置（`POKE_LOOP` 内固定か、橋渡しレイヤをどこに置くか）。
- 具体タイムアウト値（現行の rQID 同期タイムアウトと整合させる）。
- メトリクス出力先（標準出力/ファイル/外部集約）。

---

## 15. Glossary
- rQID: Showdown のリクエスト識別子。リクエストが新規到着すると増加。
- POKE_LOOP: PS クライアント/IPC 処理を担う `asyncio` イベントループ。
- publish/observe: イベント生成/待機の抽象呼称。

---

## 16. Checklist (Per-Section Resume)
- [x] S0: メトリクスフックを既存ポーリングに追加
- [x] S1: ベースライン計測を取得
- [ ] S2: Notifier 骨組み作成
- [ ] S3: Publish を受信ポンプに挿入
- [ ] S4: `_wait_for_battle_ready` をイベント化
- [ ] S5: `race_get` 微待機を撤廃
- [ ] S6: rQID 同期ループをイベント化
- [ ] S7: 失敗時のエラー停止を徹底
- [ ] S8: 事後計測を取得
- [ ] S9: ログ/メトリクスの恒久設定
