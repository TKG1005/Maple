# M4: PokemonEnv 実装 & 強化学習基本ループ用バックログ

> **Definition of Done (DoD)**
>
> * `PokemonEnv` が `gymnasium.Env` と完全互換である。
> * `reset()` / `step()` が安定動作し、複数エピソードを完走できる。
> * 非同期処理 (クライアント ↔ 環境 ↔ Showdown) でエラーが発生しない。
> * 報酬・ターン数など学習に必要な情報が収集・表示できる。

---

## 目次

1. プロジェクトフォルダ構成の整理と requirements 更新
2. `PokemonEnv` クラスファイルの作成
3. `__init__` の依存注入実装
4. `observation_space` の定義
5. `action_space` の定義
6. 対戦プレイヤーの準備とチーム読み込み
7. Showdown 対戦の非同期開始
8. 初期状態の観測と `reset` 戻り値設定
9. `step` メソッドのスケルトン追加
10. 非同期アクションキュー導入と `choose_move` 改修
11. 報酬計算ユーティリティ実装
12. エピソード終了判定の実装
13. `step` メソッドのターン同期制御
14. `step` 結果処理（観測・報酬・終了判定）
15. `render` メソッド実装
16. `close` 実装（リソース解放）
17. ランダムエージェント実装
18. 学習ループスクリプト作成 (`random_rollout.py`)
19. ログ整形 & 進捗バー追加
20. 報酬・ターン数収集と表示
21. PokemonEnv ユニットテスト追加
22. 非同期メッセージ処理テスト
23. 統合テスト – ランダム vs ダミー
24. 複数エピソード連続実行テスト
25. ドキュメント更新
26. リファクタ & コード整形
27. CI ワークフロー更新
28. 非同期設計見直し & タイムアウト
29. 依存バージョン固定 & 再検証
30. M4 完了条件の総合検証

---

## 各ステップ詳細

### 1. プロジェクトフォルダ構成の整理と `requirements.txt` 更新

* **目的**: M4 で必要となるファイル配置を決定し、依存ライブラリを明示する。
* **使用技術**: ファイル構成ベストプラクティス / パッケージ管理 (`pip`, 仮想環境)
* **達成条件**:

  * `src/`, `env/`, `agents/`, `tests/`, `scripts/` 等のディレクトリ作成。
  * `gymnasium`, `poke-env`, `numpy`, `pytest` などを `requirements.txt` へ追記。
* **テスト**: `pip install -r requirements.txt` が警告無く完了し、各種モジュールが import 可能。

---

### 2. `PokemonEnv` クラスファイルの作成

* **目的**: Gymnasium 互換環境クラスの雛形を準備。
* **使用技術**: Python クラス定義, `gymnasium.Env` 継承
* **達成条件**:

  * `src/env/pokemon_env.py` に `class PokemonEnv(gymnasium.Env)` を定義。
  * `__init__`, `reset`, `step`, `render`, `close` をダミー実装。
* **テスト**: インポートとインスタンス生成がエラーなく行える。

---

### 3. `__init__` の依存注入実装

* **目的**: 対戦相手・状態観測クラス・行動ヘルパー等を注入可能に。
* **使用技術**: 依存性注入, `numpy.random`
* **達成条件**:

  * `opponent_player`, `state_observer`, `action_helper`, `seed` を受け取り保持。
  * `self.rng = np.random.default_rng(seed)` で乱数生成器を初期化。
* **テスト**: モックを渡してインスタンス生成、属性が適切にセットされている。

---

### 4. `observation_space` の定義

* **目的**: 状態ベクトルに基づき観測空間を設定。
* **使用技術**: `gymnasium.spaces.Box`
* **Maple projectとの統合** 状態空間の次元数はMaple/src/state/state_observer.pyのget_observation_dimensionで取得可能
* **達成条件**: `self.observation_space = spaces.Box(low=0, high=1, shape=(STATE_DIM,), dtype=np.float32)`
* **テスト**: `observation_space.contains(dummy_state)` が True。

---

### 5. `action_space` の定義

* **目的**: 行動空間を設定。
* **使用技術**: `gymnasium.spaces.Discrete`
* **達成条件**: `self.action_space = spaces.Discrete(ACTION_SIZE)`
* **テスト**: `env.action_space.sample()` が常に範囲内。

---

### 6. 対戦プレイヤーの準備とチーム読み込み

* **目的**: `reset()` 時にプレイヤーを準備しチーム読み込み。
* **使用技術**: poke‑env Player API, I/O
* **達成条件**:

  * EnvPlayer と opponent\_player を生成/リセット。
  * `config/my_team.txt` からチームを読み込み設定。
* **テスト**: `env.reset()` 呼び出しで新規 Battle が生成。再呼び出しでも問題なし。

---

### 7. Showdown 対戦の非同期開始

* **目的**: ローカル Showdown サーバーで非同期対戦を開始。
* **使用技術**: poke‑env 非同期 API, `asyncio`
* **達成条件**:

  * `env.reset()` 内で EnvPlayer → opponent\_player にチャレンジ送付し、非同期で対戦開始。
  * メインスレッドをブロックしない。
* **テスト**: サーバーログに Turn 1 が表示。`reset()` がノンブロッキング。

---

### 8. 初期状態の観測と `reset` 戻り値設定

* **目的**: 初期観測値を返却。
* **使用技術**: poke‑env Battle, StateObserver
* **達成条件**:

  * Battle 取得 → `state_observer.observe(battle)` → 観測を返す。
  * `info` に `battle_tag` 等を含める。
* **テスト**: `env.reset()` が観測＋info を返し観測空間適合。

---

### 9. `step` メソッドのスケルトン追加

* **目的**: `step(action)` インターフェースを形だけ整備。
* **使用技術**: Gymnasium 規約
* **達成条件**: ダミー実装で 5 要素タプルを返す。
* **テスト**: 呼び出しで例外なし。

---

### 10. 非同期アクションキュー導入と `choose_move` 改修

* **目的**: エージェント行動をキュー経由で EnvPlayer に橋渡し。
* **使用技術**: `asyncio.Queue`, poke‑env, action\_helper
* **達成条件**:

  * `self._action_queue = asyncio.Queue()` を導入。
  * EnvPlayer.choose\_move() でキューから action\_idx を await → `action_helper.action_index_to_order()` で BattleOrder 生成。
* **テスト**: キューに値を入れると choose\_move が正しい BattleOrder を返す。

---

### 11. 報酬計算ユーティリティ実装

* **目的**: 勝敗で ±1、途中は 0。
* **使用技術**: Battle 属性参照, Pytest
* **達成条件**: `_calc_reward(battle)` 実装。
* **テスト**: 対戦を最後まで進めて で +1 / -1 / 0 を検証。

---

### 12. エピソード終了判定の実装

* **目的**: `terminated` / `truncated` 判定ロジック。エピソードの報酬結果の出力。
* **使用技術**: Battle 属性・ターン数
* **達成条件**: Battle.finished → terminated, MAX\_TURNS 超 → truncated。
* **テスト**: 対戦を1回走らせてエピソード終了を確認したログと報酬、ターン数を出力

---

### 13. `step` メソッドのターン同期制御

* **目的**: 行動送信 → 1 ターン進行を待機。
* **使用技術**: 非同期待機 (`asyncio.sleep`), Battle.turn 監視, queued_random_player.py
* **達成条件**:

  * `request` を含む最新の`rqid`のリクエストをループで待機して、リクエストが来たらエージェントに環境ベクトルを渡して、エージェントがキューにactionを投入後 、次の`request`または終了を待つ
  * 強制交代用に複数 `request` が来ても対応できること
  * `rqid`の順番が前後しても常に最新の`rqid`のリクエストに対応すること
  * タイムアウト設計。（エラーログを出力してstepが停止したことをユーザーに通知）
* **テスト**: 強制交代を含むターンでも正常に進行する。

---

### 14. `step` 結果処理（観測・報酬・終了判定）

* **目的**: 返り値 5 要素を正式実装。
* **使用技術**: StateObserver, `_calc_reward`, 終了判定
* **達成条件**: `(obs, reward, terminated, truncated, info)` を返す。
* **テスト**: ランダムエージェントでエピソード完走。

---

### 15. `render` メソッド実装

* **目的**: デバッグ用表示。
* **使用技術**: コンソール出力
* **達成条件**: Turn 情報を人間可読で表示。
* **テスト**: 任意タイミングで呼び出し→例外なし。

---

### 16. `close` 実装（リソース解放）

* **目的**: 接続・スレッドを安全に停止。
* **使用技術**: poke‑env `stop_listening`, スレッド join
* **達成条件**: `env.close()` が全リソースを解放。
* **テスト**: `close()` 後にプロセス終了でリークなし。

---

### 17. ランダムエージェント実装

* **目的**: 動作確認用の単純エージェント。
* **使用技術**: NumPy 乱数
* **達成条件**: `RandomAgent.choose_action()` が `env.action_space.sample()` を返す。
* **テスト**: 範囲外インデックスを返さない。

---

### 18. 学習ループスクリプト作成 (`random_rollout.py`)

* **目的**: エピソードを複数実行する雛形。
* **使用技術**: `argparse`, ループ制御
* **達成条件**: コマンドラインからエピソード数指定で実行可。
* **テスト**: `python random_rollout.py --episodes 3` が完了。

---

### 19. ログ整形 & 進捗バー追加

* **目的**: 実行状況の視認性向上。
* **使用技術**: `tqdm`
* **達成条件**: 進捗バーと各エピソード結果を表示。
* **テスト**: エピソード 10 で視覚確認。

---

### 20. 報酬・ターン数収集と表示

* **目的**: 学習指標の収集。
* **使用技術**: ターンカウンタ, ログ整形
* **達成条件**: 各エピソードで `reward` と `turns` をログ。
* **テスト**: 実行結果に `turns=X` が含まれる。

---

### 21. PokemonEnv ユニットテスト追加

* **目的**: 基本機能の回帰テスト。
* **使用技術**: `pytest`, モック
* **達成条件**: `tests/test_pokemon_env.py` が Green。
* **テスト**: `pytest -q` が全 PASS。

---

### 22. 非同期メッセージ処理テスト

* **目的**: メッセージ順不同耐性を検証。
* **使用技術**: artificial sleep, ログ監視
* **達成条件**:
  * 行動遅延/高速連続呼び出しでもデッドロックなし
  * `rqid` が逆転する乱序メッセージや `forceSwitch` による連続 `request`
    に正しく対応
* **テスト**: 2 秒遅延や強制交代を挟んでも正常動作。

---

### 23. 統合テスト – ランダムエージェント vs ダミー対戦相手

* **目的**: E2E 動作確認。
* **使用技術**: poke‑env RandomPlayer
* **達成条件**: 1 エピソードを安定完了。
* **テスト**: 終了フラグ確認, 報酬 ±1。

---

### 24. 複数エピソード連続実行テスト

* **目的**: 連続エピソード安定性確認。
* **使用技術**: 長時間ループ
* **達成条件**: 10 エピソード連続完走。
* **テスト**: 途中クラッシュなし・リソースリークなし。

---

### 25. ドキュメント更新

* **目的**: README / docs 反映。
* **使用技術**: Markdown
* **達成条件**: `docs/M4_setup.md` 追加 & README 更新。
* **テスト**: プレビューでリンク切れなし。

---

### 26. リファクタ & コード整形

* **目的**: 一貫性と可読性向上。
* **使用技術**: `black`, `ruff`
* **達成条件**: `black --check .` & `ruff .` がパス。
* **テスト**: 整形後もテスト Green。

---

### 27. CI ワークフロー更新

* **目的**: PokemonEnv テストを CI に統合。
* **使用技術**: GitHub Actions
* **達成条件**: CI 上で Showdown サーバ起動 → pytest 実行ジョブ追加。
* **テスト**: CI が Green。

---

### 28. 非同期設計見直し & タイムアウト

* **目的**: 無限待機防止・堅牢化。
* **使用技術**: `asyncio.wait_for`, エラーハンドリング
* **達成条件**: タイムアウト実装 & ログ追加。
* **テスト**: 行動未送信で TimeoutError が発生し、環境がリセットされる。

---

### 29. 依存バージョン固定 & 再検証

* **目的**: 再現性向上。
* **使用技術**: requirements バージョンピン
* **達成条件**: 新規環境で install → テスト Green。
* **テスト**: `pytest` 全 PASS。

---

### 30. M4 完了条件の総合検証

* **目的**: DoD 達成確認。
* **使用技術**: 総合テスト, コードレビュー
* **達成条件**: 50 エピソード連続実行で安定動作、ログ & テスト OK。
* **テスト**: 実機検証 & チームレビューで承認。

---

> **備考**
>
> * `MAX_TURNS` などのハイパーパラメータは `config/env_config.yml` などに切り出しておくと今後の実験で便利。
> * Showdown サーバーは Node.js 版をローカルで起動 (`npx pokemon-showdown`) し、ポート番号を poke‑env に渡す設定例を docs に追記すると新人でも迷いにくい。
