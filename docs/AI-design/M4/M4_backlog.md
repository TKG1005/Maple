# M4 詳細バックログ（PokemonEnv 実装とランダムエージェント検証）

以下のステップは、M4 をスムーズに進めるために **できる限り小さく分割** し、各ステップが単一の関心事に集中するように設計しています。すべてのステップはテスト可能であり、達成条件とテスト内容を明確に記述しています。

---

### Step 1: プロジェクトフォルダ構成の整理と `requirements.txt` 更新

**目的**: M4 で必要となるファイル配置を決定し、依存ライブラリを明示する。

**達成条件**

* `src/` 内に `env/`, `agents/`, `tests/`, `scripts/` サブフォルダを作成。
* `requirements.txt` に `gymnasium`, `poke-env`, `numpy`, `pytest` を追加。

**テスト内容**

* `pip install -r requirements.txt` で依存が正しくインストールできる。

**使用ライブラリ・技術**

* ファイル構成のベストプラクティス
* `pip`, `venv` または `conda`

---

### Step 2: `PokemonEnv` クラスファイルの作成

**目的**: Gymnasium 環境となるクラスのスケルトンを準備する。

**達成条件**

* `src/env/pokemon_env.py` に `class PokemonEnv(gymnasium.Env):` の宣言を追加。
* ダミーの `__init__`, `reset`, `step`, `render`, `close` を定義。

**テスト内容**

* `from env.pokemon_env import PokemonEnv` で ImportError が出ない。

**使用ライブラリ・技術**

* `gymnasium.Env` の継承パターン

---

### Step 3: `__init__` の依存注入実装

**目的**: 対戦相手プレイヤー、`StateObserver`, `ActionHelper` をコンストラクタで受け取れるようにする。

**達成条件**

* 引数でオブジェクトを受け取りインスタンス変数に保存。
* 内部乱数生成器 (`self.rng = np.random.default_rng(seed)`) を追加。

**テスト内容**

* `PokemonEnv(opponent_player, observer, action_helper)` がエラー無く生成できる。

**使用ライブラリ・技術**

* `numpy.random`, 依存注入

---

### Step 4: `observation_space` の定義

**目的**: M3 で決めた状態ベクトルの形状に基づいて観測空間を作成。

**達成条件**

* `self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(STATE_DIM,), dtype=np.float32)` を実装。

**テスト内容**

* `env.observation_space.contains(dummy_state)` が True になる。

**使用ライブラリ・技術**

* `gymnasium.spaces.Box`

---

### Step 5: `action_space` の定義

**目的**: 行動インデックス空間を Discrete として定義。

**達成条件**

* `self.action_space = gymnasium.spaces.Discrete(ACTION_SIZE)` を追加。

**テスト内容**

* `env.action_space.sample()` が 0 — `ACTION_SIZE-1` を返す。

**使用ライブラリ・技術**

* `gymnasium.spaces.Discrete`

---

### Step 6: `reset` メソッドの初期実装

**目的**: 新しいバトルを開始して初期状態を返す。

**達成条件**

* poke‑env のトレーナーをリセット or 新たに生成。
* 初期 `battle` オブジェクトから `StateObserver.observe` を呼び、状態と info を返す。

**テスト内容**

* `state, info = env.reset()` が正常に実行され、`observation_space` を満たす。

**使用ライブラリ・技術**

* `poke_env.player.Player.reset`

---

### Step 7: `step` メソッドのスケルトン追加

**目的**: 1 ターン進めるための処理フレームを作る。

**達成条件**

* 引数 `action_idx` を受け取り、`ActionHelper.action_index_to_order` に渡す。
* `battle` を 1 ステップ進め、ダミー値を返す（本実装は Step 8–10 で拡張）。

**テスト内容**

* ダミー実装でも `env.step(0)` がクラッシュしない。

**使用ライブラリ・技術**

* メソッド分割の原則

---

### Step 8: 報酬計算ユーティリティ関数実装

**目的**: 単純な勝敗ベースの報酬計算を切り出す。

**達成条件**

* `def _calc_reward(self, battle):` を定義し、`battle.finished` と `battle.won` を用いて +1 / -1 / 0 を返す。

**テスト内容**

* ユニットテストで勝利時 +1, 敗北時 -1, 途中 0 を確認。

**使用ライブラリ・技術**

* `pytest`

---

### Step 9: エピソード終了判定の実装

**目的**: `terminated`, `truncated` の正しい計算を行う。

**達成条件**

* `terminated` は `battle.finished`、`truncated` は最大ターン数超過で決定。

**テスト内容**

* ターン数 > `MAX_TURNS` で `truncated` が True になるテスト。

**使用ライブラリ・技術**

* 早期終了基準の設計

---

### Step 10: `step` メソッドの本実装

**目的**: Step 7 のスケルトンを完成させる。

**達成条件**

* 行動を送信 → バトル進行 → `StateObserver.observe` 呼び出し。
* `(state, reward, terminated, truncated, info)` を返す。

**テスト内容**

* ランダムに 10 ステップ回して期待される型が返る。

**使用ライブラリ・技術**

* poke‑env の `Player._action_to_move`

---

### Step 11: `render` と `close` のデバッグ用実装

**目的**: CLI 表示やリソース解放を行い開発デバッグを容易にする。

**達成条件**

* `render(mode="human")` で最後のターン情報を標準出力。
* `close()` で poke‑env プレイヤー接続を閉じる。

**テスト内容**

* `env.render()` が例外を投げない。

**使用ライブラリ・技術**

* 文字列フォーマット

---

### Step 12: ランダムエージェントの実装

**目的**: `action_space` に基づき行動をランダム選択できるクラスを作成。

**達成条件**

* `class RandomAgent:` に `choose_action(state)` があり `np.random.randint` で返す。

**テスト内容**

* 100 回呼び出してすべて `action_space` 範囲内。

**使用ライブラリ・技術**

* `numpy.random`

---

### Step 13: 学習ループスクリプトの作成 (`scripts/random_rollout.py`)

**目的**: Gymnasium 互換エージェント‐環境対話ループを構築。

**達成条件**

* 引数 `--episodes N` で実行回数を指定。
* 各エピソードで総報酬を計算しリストに追加。

**テスト内容**

* `python scripts/random_rollout.py --episodes 3` が完走し、合計報酬が表示される。

**使用ライブラリ・技術**

* `argparse`, ループ構造

---

### Step 14: ログ整形と簡易進捗バー追加

**目的**: 実行の可視化を改善し、長時間ロールアウトのストレスを軽減する。

**達成条件**

* `tqdm` を利用したプログレスバー。
* 各エピソード結果を `print(f"Episode {i}: reward={total_reward}")`。

**テスト内容**

* 進捗バーとログが重複せずに表示される。

**使用ライブラリ・技術**

* `tqdm`

---

### Step 15: ユニットテストの追加 (`tests/test_pokemon_env.py`)

**目的**: 主要メソッドの後方互換性を守る。

**達成条件**

* `pytest` で `reset`, `step`, `_calc_reward` の期待動作を確認。

**テスト内容**

* `pytest` 実行で green。

**使用ライブラリ・技術**

* `pytest` fixture

---

### Step 16: 統合テスト – ランダムエージェント vs ダミー対戦相手

**目的**: エンドツーエンドで Env・Agent・poke‑env が接続されるか確認。

**達成条件**

* 1 エピソードを回し、`terminated or truncated` で確実に終了する。

**テスト内容**

* スクリプトを CI（例: GitHub Actions）で実行し、30 秒以内に完了。

**使用ライブラリ・技術**

* `pytest`, GitHub Actions のワークフロー

---

### Step 17: ドキュメント更新

**目的**: README に環境構築手順と M4 の使い方を記載。

**達成条件**

* `docs/` に `M4_setup.md` を追加。
* README の「開発ステータス」に M4 完了を反映。

**テスト内容**

* Markdown リンク切れチェック。

**使用ライブラリ・技術**

* `markdownlint`（任意）

---

### Step 18: リファクタリングとコード整形

**目的**: `black`, `ruff` を用いてコードスタイルを統一。

**達成条件**

* `black --check .` と `ruff .` がエラーゼロ。

**テスト内容**

* CI で整形チェックジョブがパス。

**使用ライブラリ・技術**

* `black`, `ruff`, pre‑commit フック

---

これら 18 ステップを実施することで、M4 の完了条件（PokemonEnv 環境の実装とランダムエージェントによる動作確認）を安全に満たせます。
