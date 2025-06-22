# Maple Project M4 Backlog – PokemonEnv 実装と基本学習ループ

> **目的**
> - `PokemonEnv` を `gymnasium.Env` と互換に実装し、強化学習の基盤を整える。
> - Showdown サーバとの非同期通信を安定させ、複数エピソードを問題なく走らせる。
> - PokemonEnv_Specification.md に沿った設計・インタフェースを完成させる。

| # | ステップ名 | 目標 (WHAT) | 達成条件 (DONE) | テスト内容 (HOW) | 使用技術・ライブラリ (WITH) |
|---|-----------|-------------|----------------|-----------------|----------------------------|
| 1 | ディレクトリ整理と依存更新 | 環境実装用にフォルダ構成を整備 | `src/env/`, `src/agents/`, `scripts/` 等が作成され `requirements.txt` に `gymnasium`, `poke-env`, `numpy` などを追加 | `pip install -r requirements.txt` で警告無く終了 | Python パッケージ管理 |
| 2 | PokemonEnv 雛形追加 | Gymnasium 環境クラスを定義 | `src/env/pokemon_env.py` に `class PokemonEnv(gymnasium.Env)` を記述し、主要メソッドを空実装 | `from src.env.pokemon_env import PokemonEnv` がエラー無く通る | Python クラス定義, gymnasium |
| 3 | 依存注入の実装 | `__init__` でプレイヤー・観測クラスを受け取れるようにする | `opponent_player`, `state_observer`, `action_helper`, `seed` を保持し `np.random.default_rng(seed)` を初期化 | モック引数で生成し属性値を確認 | 依存性注入, numpy RNG |
| 4 | 観測空間の定義 | 状態ベクトル次元に基づき `observation_space` を設定 | `self.observation_space` が `Box(low=0, high=1, shape=(STATE_DIM,), dtype=np.float32)` となる | ダミー観測で `contains()` が True | gymnasium.spaces.Box |
| 5 | 行動空間の定義 | `action_space` を `Discrete(ACTION_SIZE)` とする | `env.action_space.sample()` が常に有効範囲内 | `Discrete` サンプリングで確認 | gymnasium.spaces.Discrete |
| 6 | プレイヤー生成とチーム読み込み | `reset()` 内で EnvPlayer と相手プレイヤーを初期化しチーム登録 | `config/my_team.txt` からチームを読み込み `EnvPlayer` に設定 | `env.reset()` 実行で新しい battle が生成 | poke-env, ファイル I/O |
| 7 | Showdown 対戦の非同期開始 | `play_against` を呼び非同期で対戦待機 | `env.reset()` がノンブロッキングで戻り、サーバログに Turn 1 が出る | ローカル Showdown サーバで確認 | asyncio, poke-env |
| 8 | 初期観測値の返却 | `StateObserver` を用いて最初の `observation` と `info` を返す | `reset()` の戻り値が `(obs, info)` 形式かつ観測空間に適合 | 観測内容をテストで確認 | StateObserver |
| 9 | step スケルトン実装 | `step(action)` が 5 要素タプルを返す形だけ用意 | ダミー値でも呼び出し可能 | `env.step(0)` を実行し例外無し | Gymnasium 仕様 |
|10| 非同期アクションキュー | `asyncio.Queue` を導入し `choose_move` がキューから行動を取得 | キューへインデックスを投入すると `choose_move` が対応する `BattleOrder` を返す | 単体テストでキュー処理を検証 | asyncio.Queue, action_helper |
|11| エピソード完走確認 | 対戦が最後まで実行できるか検証 | `python random_rollout.py --episodes 1` が完走する | ログに最終ターンが表示される | poke-env, asyncio |
|12| 報酬計算関数 | `_calc_reward(battle)` で勝敗に応じ ±1 を返す | 実際の対戦で +1 / -1 / 0 を確認 | 実践で確認 | poke-env Battle API |
|13| 終了判定処理 | `terminated` と `truncated` のロジック実装 | `_check_episode_end()` で `battle.finished` または `turn > MAX_TURNS` を判定 | 実戦でフラグ確認 | Battle 属性参照 |
|14| ターン同期機構 | `_race_get` を用いて最新 `request` を取得 | `rqid` 乱序でもターンが進む | 強制交代シナリオで確認 | asyncio, rqid 管理 |
|15| step 出力整備 | observations, rewards, terminated, truncated, infosを player識別子(`player_0`,`player_1`)をキーとしたdictで返す | ランダムエージェントで 1 戦完走 | `run_battle.py` 実行 | 全機能統合 |
|16| render 実装 | ターン情報をコンソール表示 | `env.render()` を呼んでも例外なし | 目視確認 | ロギング |
|17| close 実装 | WebSocket とキューを安全に閉じる | `env.close()` 後にリーク無し | プロファイル確認 | poke-env `stop_listening` |
|18| MapleAgent ベースライン | 乱択行動を返すシンプルエージェント | マスク内インデックスのみ選択 | ユニットテスト | numpy RNG |
|19| バトル実行スクリプト | `run_battle.py` で複数戦を処理 | `--n 3` で完走 | CLI 実行 | argparse, tqdm |
|20| 進捗バー付きログ | `tqdm` で進捗バーを表示 | 10 戦実行でバーが動く | 実行結果を目視確認 | tqdm |
|21| 戦績ログ集計 | 各戦の報酬とターン数をまとめる | 結果ログに平均値を出力 | スクリプト実行結果確認 | logging |
|22| PokemonEnv ユニットテスト | reset/step/close の基本動作を検証 | `pytest -q` が PASS | 自動テスト | pytest |
|23| 非同期メッセージテスト | メッセージ遅延や順序入替に耐性確認 | artificial sleep を挿入してもデッドロックしない | 遅延シナリオで確認 | asyncio, poke-env |
|24| E2E 統合テスト | ランダムエージェント同士で 1 戦実施 | 終了フラグと報酬 ±1 を確認 | スクリプトで対戦完了 | poke-env RandomPlayer |
|25| 複数エピソード試験 | 10 連戦以上でも安定動作 | `--n 10` で完走し例外無し | 長時間テスト | |
|26| ドキュメント更新 | M4 セットアップ手順を記述 | `docs/M4_setup.md` 追加済み | Markdown リンクチェック | Markdown |
|27| コード整形 & リファクタ | `black` と `ruff` を適用 | `black --check .` `ruff .` ともに PASS | フォーマット後テスト実行 | black, ruff |
|28| CI ワークフロー更新 | GitHub Actions で自動テスト | CI が緑になる | PR 上で確認 | GitHub Actions |
|29| タイムアウト処理改善 | `asyncio.wait_for` を適用してハング防止 | 行動せず待機すると TimeoutError 発生 | 専用テスト | asyncio.wait_for |
|30| 依存バージョン固定 | `requirements.txt` にバージョン指定 | 新規環境で `pytest` 全て PASS | インストール確認 | pip version pinning |
|31| M4 完了レビュー | Backlog 完了を総合確認 | 50 戦連続で安定しレビュー承認 | 実機テスト + レビュー | 総合確認 |

> **備考**
> - `PokemonEnv_Specification.md` のフロー図を常に参照して実装を進めること。
> - `MAX_TURNS` 等のパラメータは `config/env_config.yml` にまとめると管理しやすい。
> - Showdown サーバは `npx pokemon-showdown` でローカル起動しポート `8000` を利用する想定。
> - ステップ10完了後、エピソードが最後まで完了することを確認してから報酬計算と終了判定を実装する。
