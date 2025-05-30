# Maple Project – Functional Requirements

本書は Maple フレームワークにおける強化学習環境 **PokemonEnv** の機能要件を整理したものです。AI エンジニアリング LLM が実装・テストを行う際の指針となるよう、Gym API 準拠・通信仕様・状態／行動空間・報酬設計などの必須要件を中心にまとめています。

---

## 目次

1. 対象範囲
2. 用語定義
3. 機能要件一覧

   1. Gymnasium API
   2. 状態観測 (StateObserver)
   3. 行動空間 (ActionHelper)
   4. 通信 & 同期処理
   5. 報酬設計
   6. エピソード管理
4. 非機能的制約 (抜粋)
5. テスト基準

---

## 1. 対象範囲

* RL 環境 `PokemonEnv` とその周辺モジュール (`_AsyncPokemonBackend`, `EnvPlayer`, `StateObserver`, `ActionHelper`) を含む。
* 対戦相手 `OpponentPlayer` は poke‑env の既存実装 (例: `RandomPlayer`) を利用する前提。
* Showdown サーバはローカル起動済みとし、poke-env の `LocalhostServerConfiguration` による接続を行う。

---

## 2. 用語定義

| 用語               | 意味                                                                              |
| ---------------- | ------------------------------------------------------------------------------- |
| **Episode**      | 1 試合 (6→3 シングル) の対戦を開始から勝敗決着まで実行する一連のステップ。                                      |
| **Step**         | 1 ターン (自分と相手が行動を選択し結果が確定するまで) を環境視点で進める最小単位。Gym `step()` 1 回に対応。                |
| **Observation**  | `StateObserver.observe()` により得られる `numpy.ndarray` の状態ベクトル。                      |
| **Action Index** | `ActionHelper` が定義する離散行動インデックス (0–9)。                                           |
| **BattleOrder**  | poke‑env が Showdown へ送るコマンドオブジェクト。`ActionHelper.action_index_to_order()` により生成。 |
| **Terminated**   | 勝敗確定・引き分けなど通常終了した場合のフラグ。Gym `terminated=True`。                                  |
| **Truncated**    | タイムアウト・通信切断など異常終了した場合のフラグ。Gym `truncated=True`。                                 |

---

## 3. 機能要件一覧

### 3.1 Gymnasium API

| 番号  | 要件                                                                                              | 重要度        |
| --- | ----------------------------------------------------------------------------------------------- | ---------- |
| G-1 | `PokemonEnv` は `gymnasium.Env` を継承し、`reset`, `step`, `close`, `render` を実装すること。                 | **Must**   |
| G-2 | `reset()` は `(obs: ndarray, info: dict)` を同期的に返すこと。                                             | **Must**   |
| G-3 | `step(action_idx)` は `(obs, reward, terminated, truncated, info)` を同期的に返すこと。                    | **Must**   |
| G-4 | `observation_space` と `action_space` は Gym 仕様に従い事前定義し、`spaces.Box` と `spaces.Discrete` を使用すること。 | **Must**   |
| G-5 | `render(mode="human")` はテキストベースの簡易出力で良いが、現在ターン・両者 HP・利用技 PP を表示すること。                            | **Should** |

### 3.2 状態観測 (StateObserver)

| 番号  | 要件                                                                               | 重要度        |
| --- | -------------------------------------------------------------------------------- | ---------- |
| S-1 | `StateObserver` は `Battle` オブジェクトを受け取り、固定長 `numpy.ndarray(dtype=float32)` を返すこと。 | **Must**   |
| S-2 | 観測ベクトルの次元数は `get_observation_dimension()` で取得可能であること。                            | **Must**   |
| S-3 | 特徴量定義を YAML (`config/state_spec.yml`) から読み取り、ポケモン種別や技スロット増減に依存しない汎用設計とすること。      | **Should** |
| S-4 | 観測には最低限以下を含める: 自ポケモン HP%, 相手 HP%, 残ポケ数, 天候, フィールド, 各技 PP 残量フラグ。                  | **Must**   |

### 3.3 行動空間 (ActionHelper)

| 番号  | 要件                                                                                | 重要度      |
| --- | --------------------------------------------------------------------------------- | -------- |
| A-1 | 行動スペースサイズ `ACTION_SPACE_SIZE` を 10 とし、インデックス割当は 0–3=技(通常), 4–7=テラスタル技, 8–9=交代とする。 | **Must** |
| A-2 | `get_available_actions(battle)` は長さ 10 のバイナリマスクを返し、選択不可の行動は 0 とする。                | **Must** |
| A-3 | `action_index_to_order(player, battle, idx)` は無効インデックス入力で `ValueError` を投げること。    | **Must** |
| A-4 | 定義外の技スロットや交代先が存在しない場合でもマスクが 0 となることで安全に弾くこと。                                      | **Must** |

### 3.4 通信 & 同期処理

| 番号  | 要件                                                                                                            | 重要度        |
| --- | ------------------------------------------------------------------------------------------------------------- | ---------- |
| C-1 | poke‑env (`EnvPlayer` / `OpponentPlayer`) は WebSocket で Showdown サーバへ接続し、`start_listening=True` で自動受信を開始すること。 | **Must**   |
| C-2 | `PokemonEnv.reset()` 中に両プレイヤーのログイン完了 (`wait_for_login`) を await すること。                                         | **Must**   |
| C-3 | `Backend` は `RESET_TIMEOUT = 30s` までに `_current_battle` が確立しない場合、例外を送出してエピソードをトランケートすること。                     | **Must**   |
| C-4 | 各 `step()` は `STEP_TIMEOUT = 10s` 以内にターン結果を受信できない場合、警告を出し `truncated=True`・`terminated=True` として返すこと。         | **Must**   |
| C-5 | ShowdownException や WebSocket 切断時は対戦を即時終了し、`reward = REWARD_LOSS` とすること。                                      | **Must**   |
| C-6 | 強制交代（`forceSwitch`）を検出した場合は `available_switches[0]` を自動選択し、タイムアウトを防ぐこと。                                       | **Should** |

### 3.5 報酬設計

| 番号  | 要件                                                                              | 重要度        |
| --- | ------------------------------------------------------------------------------- | ---------- |
| R-1 | 勝利時 `REWARD_WIN = +1.0`, 敗北時 `REWARD_LOSS = -1.0`, 引き分け `REWARD_TIE = 0.0` とする。 | **Must**   |
| R-2 | 無効行動ペナルティ `REWARD_INVALID = -0.01` を即時付与し、エピソードは継続させること。                        | **Must**   |
| R-3 | 報酬は float (32 bit) で返すこと。                                                       | **Should** |

### 3.6 エピソード管理

| 番号  | 要件                                                                            | 重要度        |
| --- | ----------------------------------------------------------------------------- | ---------- |
| E-1 | `terminated=True` でエピソード終了を示す。勝敗が決したターンに設定すること。                               | **Must**   |
| E-2 | `truncated=True` はタイムアウトまたは通信障害による異常終了でのみ用いること。                               | **Must**   |
| E-3 | エピソード終了後の `step()` 呼び出しはエラーを返すか、`done=True` のまま無報酬で返すこと。                      | **Should** |
| E-4 | `reset()` はエピソード間で必要な内部状態 (`_current_battle`, `battle_is_over` 等) を完全初期化すること。 | **Must**   |

---

## 4. 非機能的制約 (抜粋)

* **実行速度**: 1 ターンあたり 200 ms 以内で `step()` 戻り値を出せること (ネットワーク遅延除く)。
* **再現性**: `reset(seed)` で乱数シードを指定した場合、エージェント乱数・ポケモン選出順・ActionHelper のランダム選択順も決定論的となること。
* **ロギング**: Python 標準 `logging` を用い、ライブラリ内部ではロガー階層に依存せず利用者側で `basicConfig` を調整可能とすること。

---

## 5. テスト基準

| テスト ID | 観点      | 合格基準                                                                                    |
| ------ | ------- | --------------------------------------------------------------------------------------- |
| T-1    | スモークテスト | `tests/test_env_step_loop.py` が 3 エピソード連続でクラッシュせず完走し、各ステップで `obs` shape と `done` 型が正しい。 |
| T-2    | 無効行動    | 行動インデックス 99 を入力し、`reward=-0.01`, `done=False` で返る。                                      |
| T-3    | タイムアウト  | Showdown サーバを意図的に停止→ `step()` が 10s で戻り `truncated=True` になる。                           |
| T-4    | 勝利/敗北   | モック Opponent を設定し、自分勝利時に `reward=+1.0`, 敗北時 `-1.0` で終了。                                 |
| T-5    | 観測次元    | `StateObserver.get_observation_dimension()` と `obs.shape[0]` が一致する。                     |

---

### 備考

* 非機能要件 (ログ、メモリ使用量、並列化) は別ドキュメントで詳細化予定。ここでは最低限の制約のみ記載。
* 実装者は本要件に準拠しつつ、必要に応じ拡張ポイントを反映して柔軟に設計を行ってください。
