# Maple Project – ステップバイステップタスク一覧

> 各タスクは **小さくテスト可能**・**開始と終了が明確**・**単一の関心ごと** に集中しています。
> タスク ID は「Px‑Tx」の形式で、P はフェーズ（マイルストーン）、T は通番です。

---

## P0 基盤セットアップ（M1）

| ID    | タスク                  | 開始条件            | 完了条件（テスト方法）                                         |
| ----- | -------------------- | --------------- | --------------------------------------------------- |
| P0‑T1 | Poetry プロジェクト初期化     | リポジトリが空 or 最小構成 | `poetry install` が依存関係を解決し、`pytest -q` が 0 テストで成功終了 |
| P0‑T2 | pre‑commit Hooks 導入  | P0‑T1 完了        | `pre‑commit run --all-files` がエラー無く完了               |
| P0‑T3 | GitHub Actions CI 雛形 | P0‑T2 完了        | プッシュ時に「Install → Lint → Test」3 ジョブが緑になる             |

---

## P1 ユーティリティ実装（M1）

| ID    | タスク                        | 開始条件  | 完了条件                                                                    |
| ----- | -------------------------- | ----- | ----------------------------------------------------------------------- |
| P1‑T1 | `maple/utils/logger.py` 作成 | P0 完了 | `pytest tests/test_logger.py` が PASS（フォーマット & 出力レベル確認）                  |
| P1‑T2 | `config.py` で Dataclass 設計 | P1‑T1 | `python -c "from maple import config; print(config.Settings())"` が属性を表示 |
| P1‑T3 | `registry.py`（簡易 DI）       | P1‑T2 | テストでクラス登録→取得が成功                                                         |

---

## P2 poke‑env ホスト層（M2）

| ID    | タスク                      | 開始条件  | 完了条件                                           |
| ----- | ------------------------ | ----- | ---------------------------------------------- |
| P2‑T1 | `showdown_host.py` スタブ作成 | P1 完了 | Unit テストで `LocalhostServer` が起動し `/ping` 応答を返す |
| P2‑T2 | Battle モック生成ユーティリティ      | P2‑T1 | テストで `MockBattle()` が HP, moves を持つ            |

---

## P3 状態 & 行動空間（M3）

| ID    | タスク                    | 開始条件  | 完了条件                                                  |
| ----- | ---------------------- | ----- | ----------------------------------------------------- |
| P3‑T1 | `state_observer.py` 雛形 | P2 完了 | `observe(mock_battle)` が ndarray を返し shape=(N,) をアサート |
| P3‑T2 | `action_helper.py` 雛形  | P3‑T1 | `index_to_order(mock_battle, idx)` が BattleOrder を返す  |
| P3‑T3 | 有効行動マスク関数追加            | P3‑T2 | ランダム idx が無効なら ValueError を投げるテストが PASS               |

---

## P4 Gymnasium 環境（M4）

| ID    | タスク                           | 開始条件  | 完了条件                                                                |
| ----- | ----------------------------- | ----- | ------------------------------------------------------------------- |
| P4‑T1 | `PokemonEnv` コンストラクタ実装        | P3 完了 | `env = PokemonEnv(...); assert env.observation_space.shape == (N,)` |
| P4‑T2 | `reset()` 実装                  | P4‑T1 | `state, info = env.reset(); assert state.shape == (N,)`             |
| P4‑T3 | `step()` 実装（報酬: win/lose 基本形） | P4‑T2 | ランダム行動エピソード 1 で `terminated in {True,False}` を確認                    |
| P4‑T4 | `render()` & `close()` スタブ    | P4‑T3 | 呼び出しても例外が起きない                                                       |
| P4‑T5 | 統合テスト `test_env_step_loop.py` | P4‑T4 | 10 エピソードを実行し全て正常終了ログ                                                |

---

## P5 エージェント基盤（M4→M5）

| ID    | タスク                          | 開始条件  | 完了条件                                            |
| ----- | ---------------------------- | ----- | ----------------------------------------------- |
| P5‑T1 | `BaseAgent` プロトコル定義          | P4 完了 | `isinstance(RandomAgent(), BaseAgent)` テスト PASS |
| P5‑T2 | `RandomAgent` 実装             | P5‑T1 | 100 ステップで常に `action_space.contains(a)` True     |
| P5‑T3 | エピソードループ (`train_random.py`) | P5‑T2 | CLI `--episodes 5` で総報酬が CSV 出力                 |

---

## P6 強化学習ループ（M5）

| ID    | タスク                         | 開始条件  | 完了条件                                            |
| ----- | --------------------------- | ----- | ----------------------------------------------- |
| P6‑T1 | Experience Replay Buffer 実装 | P5 完了 | unit test で push/pop サイズ一致                      |
| P6‑T2 | DQN ネットワーク定義 (PyTorch)      | P6‑T1 | forward に対し `out.shape == (batch, action_size)` |
| P6‑T3 | `DQNAgent` with ε‑greedy    | P6‑T2 | 1 ステップで replay 写真が追加される                         |
| P6‑T4 | `train_rl.py` 初期ループ実装       | P6‑T3 | 学習 1 epoch 実行し loss が float でログ出力               |
| P6‑T5 | モデル保存 & ロード                 | P6‑T4 | ファイル生成→ロード後 `state_dict` 同値                     |

---

## P7 自己対戦 & 並列化（M6）

| ID    | タスク               | 開始条件  | 完了条件                              |
| ----- | ----------------- | ----- | --------------------------------- |
| P7‑T1 | Self‑Play 対戦マネージャ | P6 完了 | 2 つの Agent で並列対戦し result JSON が生成 |
| P7‑T2 | 並列学習 (Ray) PoC    | P7‑T1 | ワーカー 2→4 でステップスループが倍増             |

---

## P8 評価 & 可視化（M7）

| ID    | タスク                    | 開始条件  | 完了条件                               |
| ----- | ---------------------- | ----- | ---------------------------------- |
| P8‑T1 | `evaluate_agent.py` 実装 | P7 完了 | 100 戦して勝率が float で表示               |
| P8‑T2 | TensorBoard ログ出力       | P8‑T1 | `tensorboard --logdir runs` でグラフ確認 |

---

## P9 Switch 実機ビジョン PoC（M9）

| ID    | タスク               | 開始条件     | 完了条件            |
| ----- | ----------------- | -------- | --------------- |
| P9‑T1 | 画面キャプチャ取得スクリプト    | ハードウェア接続 | PNG が保存される      |
| P9‑T2 | YOLOv8 で HP/技 OCR | P9‑T1    | 画像→JSON 情報抽出が成功 |

---

## P10 オンライン対戦統合（M10）

| ID     | タスク              | 開始条件             | 完了条件                             |
| ------ | ---------------- | ---------------- | -------------------------------- |
| P10‑T1 | Showdown ログイン自動化 | オンライン API アクセス許可 | `login_success` フラグ True         |
| P10‑T2 | リアルタイム戦況ストリーム    | P10‑T1           | `websocket` で battle messagesを受信 |

---

### 補足

* すべてのタスクは **テスト駆動** (pytest) 推奨。
* `docs/spec.md` に合わせて更新が必要な場合はドキュメント更新タスクを追加してください。
* フェーズ間の依存を最小化するために、インターフェースを早期に確定させること。

---

作成: 2025‑05‑27
