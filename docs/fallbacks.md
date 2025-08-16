# フォールバック処理一覧（online/local 共通 + モード別）

本ドキュメントは、train.py と関連コードに実装されているフォールバック（代替処理／安全側挙動）を網羅的に整理したものです。学習継続のためのランダム化・デフォルト値・警告継続と、診断重視のフェイルファスト（停止）を区別して記載します。

---

## デバイス・初期化・ログ
- デバイス選択（`src/utils/device_utils.py`）
  - `--device cuda` で CUDA 不可 → CPU へフォールバック（★ 環境差・数値非決定性により学習結果が変わり得る）
  - `--device mps` で MPS 不可 → CPU へフォールバック（★ 同上）
  - 無効なデバイス文字列 → CPU へフォールバック（★ 同上）
- 種族埋め込み（`src/agents/species_embedding_layer.py`）
  - `pokemon_stats.csv` 読み込み失敗 → 埋め込みはランダム初期化（★ 初期条件が変わり学習結果に影響）
- poke-env ログパッチ（`src/utils/poke_env_logging_fix.py`）
  - poke-env 未インポート等で失敗 → パッチ適用せず継続（標準ログ）

---

## チーム・設定・サーバ
- チームローディング（`src/teams/`）
  - 0件/空ファイル → `get_random_team()` は `None`（上位で default へ）
  - チームディレクトリ不存在/空 → 空集合と0統計で返却（警告）
- チーム選択（`PokemonEnv._get_team_for_battle`）
  - random で0件 → default チームへフォールバック（警告）
  - default ファイル不存在 → `None` 返却（警告、後段で失敗の可能性を許容）
- 環境設定（`PokemonEnv.__init__`）
  - `env_config.yml` 読み込み失敗 → `timeout=5.0` にフォールバック
  - local で `server_configuration=None` → `ServerConfiguration('localhost', 8000)` にフォールバック
- サーバ（`src/utils/server_manager.py`）
  - サーバ設定無し → `localhost:8000` 単一構成にフォールバック
  - 容量超過（parallel > capacity） → 例外（フォールバックなし）

---

## 行動選択（エージェント）
- `RLAgent.select_action`（`src/agents/RLAgent.py`）
  - 有効マスク0件 → 全行動一様分布で確率生成（確率フォールバック）（★ 探索分布が変わり軌跡・学習に影響）
- `MapleAgent.select_action`（`src/agents/MapleAgent.py`）
  - 有効インデックスなし → `env.action_space` からランダムサンプル（ランダム行動）（★ 軌跡・学習に影響）
- `EpsilonGreedyWrapper`（`src/agents/action_wrapper.py`）
  - 有効行動なし → 内包エージェントの返り値に委譲（警告ログのみ）
  - 混合確率合計0 → 有効行動の一様分布に正規化（★ 行動分布が変わり学習に影響）
- `RandomAgent`（`src/agents/random_agent.py`）
  - 有効行動なし → `0` を返す（デフォルト行動）（★ 軌跡・学習に影響）

---

## 行動マッピング・命令変換（`src/action/action_helper.py`）
- 詳細生成で対象不在 → 「Switch to position N」等の汎用表示にフォールバック
- `action_index_to_order[_from_mapping]`
  - move 不在かつ `sub_id == 'struggle'` → `move struggle` にフォールバック
  - テラスタル不可（`battle.can_tera is None`） → 通常技として送信
  - switch マッピング情報不足 → 位置指定 `/choose switch <position>` を直接生成

---

## rQID 同期・アクション投入（共通：online/local）
- 数値アクションの rQID 不整合（`PokemonEnv._process_actions_parallel`）
  - 不一致 → 当該アクションはドロップして再待機（代替行動には置換しない）
- rQID 更新待ち（`PokemonEnv.step`）
  - 前回 rQID と同一/未設定 → `timeout` までポーリング
  - 期限超過 → `[RQID TIMEOUT]` をログし例外（フェイルファスト）
- `need_action` ゲート（`PokemonEnv.step`）
  - `wait/teampreview` 時の数値アクション → ドロップ（誤送信防止）
- 終了後の残キュー → ベストエフォートでドレインして `join()` ブロック回避

---

## IPC（local モード）
- 送信経路（`EnvPlayer._send_battle_message_local_aware`）
  - local 時のコマンド正規化（`/team`→`team`、`/choose` 前置削除）でプロトコル互換にフォールバック
  - local 経路が無い/失敗 → WebSocket 経路へ送信
- IPC コントローラ（`src/env/ipc_battle_controller.py`）
  - Node からの `battle_closed` → busy フラグ解除（再利用）
  - 不正 JSON/制御メッセージ（error/exit） → エラーログのみで継続
- Node ブリッジ（`scripts/node-ipc-bridge.js`）
  - `|request|/|init|` の rqid 欠落 → 連番で注入
  - `|win|/|tie|` → セッション自動削除 + `battle_closed` 通知
  - JSON 化/送信失敗 → stderr にエラー出力（継続）
- IPC プール（`src/env/controller_registry.py`）
  - 上限到達（max_processes） → 先頭コントローラを返す（簡易フォールバック）

---

## WebSocket（online モード）
- local 条件を満たさない場合は常に WS 経路で送信（local→WS の暗黙的フォールバック）

---

## 学習オーケストレーション（`train.py`）
- 行動 API 呼び分け 
  - `select_action` 未実装 → `act()` にフォールバック
- 対戦相手アクション
  - `env._need_action[opponent] == False` → `act1=0`（デフォルト行動）
- 自己対戦/履歴相手
  - 歴史対戦相手選択に失敗 → 現行スナップショットを使用（★ 対戦相手分布が変わり学習に影響）
  - `--opponent_mix/--opponent` 未指定 → `self`（自己対戦）（★ 対戦相手分布が変わり学習に影響）
- スケジューラ
  - 無効設定 → スケジューラなし（現行 LR 維持）（★ 学習率経路が変わり学習に影響）
- ε-greedy 統計
  - ラッパが `get_exploration_stats` 非実装 → 警告ログのみ（学習継続）

---

## フェイルファスト（フォールバックなし）
- rQID 更新不能（`PokemonEnv.step`） → タイムアウトで例外
- 無効インデックス/無効状態の `action_index_to_order*` → 例外
- Node/IPC 致命的異常（`_fatal_stop`） → 切断/例外
- MultiServerManager 容量超過 → 例外

---

## 要点
- 行動面は「有効行動なし」に対してランダム/一様分布を複数箇所で実装（継続性重視）。
- 通信・同期（特に rQID）は「捨てて待つ→期限超過で停止」という診断優先の設計。
- local（IPC）はプロトコル整合（コマンド正規化・rqid注入）とセッション自動削除で健全性を維持。
- 設定・資源は適切なデフォルトへ丸めて継続（デバイス/サーバ/チーム/ログ）。
- online への影響は分離実装で最小化（local専用フォールバックは online に干渉しない）。
