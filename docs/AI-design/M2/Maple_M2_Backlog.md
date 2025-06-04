# Maple Project M2 Backlog – 基本対戦実行可能

> **目的**  
> ― ローカル Showdown サーバ上で、`poke-env` を通じて Python クライアント同士が 1 戦完走できる状態をつくる。  
> ― **強化学習や高度なアルゴリズムはまだ扱わず**、まずは対戦フローの”往復”を確認する。

| # | ステップ名 | 目標（WHAT） | 達成条件（DONE の定義） | テスト内容（HOW） | 主要ライブラリ／技術（WITH） |
|---|------------|--------------|-------------------------|-------------------|------------------------------|
| 1 | 開発用仮想環境の作成 | 依存関係を切り分ける | `poetry env info` または `python -m venv venv` に成功し、`pip list` が空に近い | 仮想環境内で `python -V` が期待バージョン (例: 3.11) を返す | Python, venv / poetry |
| 2 | `requirements.txt` 雛形作成 | ライブラリの一覧を管理 | ファイルに `poke-env`, `gymnasium`, `websockets`, `python-dotenv` などが列挙され Git 管理に追加 | `pip install -r requirements.txt` が警告なく完了 | pip |
| 3 | Node.js & Showdown サーバ取得 | 対戦ホストを準備 | `node -v` が表示され、`showdown` コマンドでヘルプが出る | `npx pokemon-showdown` 起動後、`localhost:8000` ブラウザ表示成功 | Node.js, Showdown |
| 4 | Showdown サーバ設定ファイル編集 | gen9ou シングル 6→3 用 | `config/custom-formats.js` に `"gen9ou"` が存在し、`config/config.js` で `port = 8000` など反映 | `npx pokemon-showdown validate-format gen9ou` が OK を返す | Showdown 設定スクリプト |
| 5 | サーバ起動スクリプト作成 | 再起動を自動化 | `scripts/start_server.sh` が `chmod +x` 済みで `./scripts/start_server.sh` が Node プロセスを前景起動 | プロセスが 1 回起動 → CTRL-C で正常終了 | Bash |
| 6 | `my_team.txt` 作成 | 手動チームを登録 | 6 匹分の Showdown 形式テキストが保存 | `poke-env` の `to_showdown_format` で再変換し一致 | poke-env |
| 7 | シンプル対戦クライアント骨格 | `SimpleAgent` クラスを実装 | `python simple_agent.py --team my_team.txt` でインスタンス生成のみ成功 | インポート時エラーゼロ、`print(agent.team)` 表示 | Python OOP, poke-env |
| 8 | ランダム AI 同士の 1 戦スクリプト | 1 本対戦を完走させる | `python run_battle.py` が終端コード 0 で終了し、勝敗が標準出力に表示 | バトルログ中に `"Win"` または `"Loss"` が含まれる | poke-env (`RandomPlayer`), asyncio |
| 9 | 勝敗・ターン数の取得ロジック | 結果をプログラムで参照 | `run_battle.py` 実行後、logger で `{winner:"p1", turns:25}` など出力 | `assert result["turns"] > 0` を含むテストが通る | Python 標準 `logging` |
|10 | 連続バトル 10 回実行 | 安定して複数試合 | `python run_battle.py --n 10` が 10 試合完走、平均ターン数を表示 | 失敗 (例: 接続切れ) が 0 件 | asyncio, tqdm |
|11 | Gymnasium 環境ラッパー雛形 | 後続 RL の足場 | `PokemonBattleEnv` クラス (Gymnasium Env) が `reset()` と `step()` をダミー実装 | `env = PokemonBattleEnv()` で `env.action_space`, `env.observation_space` が出力 | gymnasium |
|12 | README 更新 | 手順をドキュメント化 | `README.md` にローカルで 1 戦するまでのコマンド列が箇条書き | 新規環境で README 通り実行し btl 完了 | Markdown |
|13 | GitHub Actions (option) | CI でインポート・テスト | `.github/workflows/test.yml` が `pytest` を実行 | PR ごとに CI 緑 | GitHub Actions, pytest |

#### 補足・次の見通し  
- **M2 の完了判定**: 上記 #1〜#10 (+README) が通り、「コマンド一発でローカル対戦が終わる」状態。  
