# Maple Project M6 Backlog – 自己対戦学習環境構築

## 目的
- **PokemonEnv** 上で複数エージェント（モデル）による自己対戦を **安定動作 & 長時間実行** できるようにする  
- 対戦環境の **安定性・拡張性・自動実行性・高速性** を高め、学習アルゴリズム改良 (M7) が乗る基盤を構築  
- PokemonEnv 技術仕様書準拠の **マルチエージェント API**（`dict[player_id]` で観測・行動を管理）を守り、**ランダム AI / 学習済みモデル** など多様な組み合わせに対応  

---

## バックログ（全 19 ステップ）

> 各ステップは「Goal (WHAT) ▸ Done (DONE) ▸ Test (HOW) ▸ Tech (WITH)」で整理しています。

### Step 1 : 自己対戦学習スクリプト雛形作成
- **Goal** `train_selfplay.py` を新規作成し、環境 & エージェント初期化フローを実装  
- **Done** `python train_selfplay.py --dry-run` で環境だけ初期化して即終了  
- **Test** スクリプト内で `PokemonEnv()` をインスタンス化後、正常終了すること  
- **Tech** Gymnasium / PokemonEnv / Python

---

### Step 2 : 2エージェント初期化
- **Goal** `PokemonEnv` をマルチエージェントモードで起動し、`player_0`・`player_1` 用に 2 × `RLAgent` を生成  
- **Done** 両 `RLAgent` がエラーなく初期化・モデル保持  
- **Test** `env.agent_ids == ("player_0","player_1")` にそれぞれ紐付くか検証  
- **Tech** PokemonEnv / RLAgent

---

### Step 3 : 重み共有オプション
- **Goal** 両プレイヤーが同じ **PolicyNetwork** を共有する `--shared-policy` モードを実装  
- **Done** 共有時、片方で重み更新 ⇒ もう片方にも反映  
- **Test** オブジェクト ID が一致するか・学習 1 step 後に推論結果が変わるか確認  
- **Tech** PyTorch / PolicyNetwork

---

### Step 4 : チームプレビュー処理実装
- **Goal** 対戦開始前の **Team Preview** に対応し、`choose_team` を呼び出す  
- **Done** `info["request_teampreview"] == True` 時に 3 体選出し、`env.step()` に渡る  
- **Test** `--dry-run` でチーム選択が発生・初期観測を取得できること  
- **Tech** poke-env (EnvPlayer) / MapleAgent (`choose_team`)

---

### Step 5 : ターンごとの行動ループ実装
- **Goal** `while not done:` ループで両エージェントが行動を選択し `env.step()`  
- **Done** ランダム AI 同士などで 1 エピソード完走  
- **Test** 行動が毎ターン環境へ渡り、終了時 `done == True`  
- **Tech** Gymnasium ループ同期処理

---

### Step 6 : 遷移保存とリプレイバッファ
- **Goal** (観測, 行動, 報酬, 次状態, 終了) を経験リプレイに蓄積（両プレイヤー分）  
- **Done** ターン数 N の対戦で **2 × N 件** の遷移が保存  
- **Test** `len(replay_buffer)` が期待値に一致し、`player_0/1` のデータ構造を持つ  
- **Tech** NumPy / ReplayBuffer クラス

---

### Step 7 : ネットワーク更新処理
- **Goal** バッファからサンプルして **PolicyNetwork** を更新  
- **Done** 学習後にパラメータ変化・損失が減少傾向  
- **Test** 学習前後で予測が変わるか／損失ログ確認  
- **Tech** PyTorch (autograd / optimizer) / NumPy

---

### Step 8 : エピソード反復と終了条件
- **Goal** `--episodes N` 指定で N エピソード自己対戦を繰り返し、終了時に環境を `close()`  
- **Done** 例: `--episodes 5` で 5 戦実行・終了時に全資源解放  
- **Test** バックグラウンド接続が残らないか確認  
- **Tech** Python / PokemonEnv `.close()`

---

### Step 9 : 自己対戦設定項目追加
- **Goal** ハイパーパラメータ (エピソード数・学習率・`--shared-policy` 等) を CLI / YAML で設定可能に  
- **Done** `python train_selfplay.py --help` で主要オプションが表示  
- **Test** 引数変更で動作が切替わるか検証  
- **Tech** argparse / 設定ファイル

---

### Step 10 : モデル保存処理
- **Goal** 学習後に重みを `best_model.pt` などで保存・再利用可能に  
- **Done** `--save best_model.pt` でファイル生成、再ロードで対戦 OK  
- **Test** `evaluate_rl.py` で読み込み対戦  
- **Tech** PyTorch `torch.save / load`

---

### Step 11 : チェックポイント保存
- **Goal** 長時間学習向けに `--checkpoint-interval K` ごと保存 (`checkpoints/ checkpoint_10.pt` …)  
- **Done** 間隔通りにファイル増加・再開可能  
- **Test** 中間モデルで `evaluate_rl.py` が動作  
- **Tech** PyTorch / ファイル I/O

---

### Step 12 : 自己対戦スクリプト自動テスト
- **Goal** CI (pytest) で **1 エピソード学習** が PASS するケースを追加  
- **Done** `pytest -k test_train_selfplay_one_episode` が成功  
- **Test** `subprocess` で `train_selfplay.py --episodes 1` 実行・コード 0  
- **Tech** pytest / GitHub Actions

---

### Step 13 : 二エージェント対戦統合テスト
- **Goal** ランダム NN 方策 2 体で自己対戦し、終了判定 & 報酬を検証  
- **Done** 勝者:+1 / 敗者:-1 (引分:0)、`terminated` フラグが正しく設定  
- **Test** 各 dict (`rewards`, `terminated`) の一貫性をアサート  
- **Tech** PokemonEnv / RLAgent

---

### Step 14 : 異種エージェント対戦検証
- **Goal** 例: 学習済みモデル vs ランダムエージェント を検証  
- **Done** 双方の行動ログが記録され、対戦結果取得  
- **Test** `evaluate_rl.py --opponent random` で 5 戦完走  
- **Tech** PokemonEnv / MapleAgent / RLAgent

---

### Step 15 : 並列対戦環境対応
- **Goal** スレッド・プロセス・Gymnasium VectorEnv などで複数環境を並列実行  
- **Done** 2 並列で独立対戦し、シングルより実行時間短縮  
- **Test** 壁時計時間を比較し、速度向上を確認  
- **Tech** Multithreading / VectorEnv / WebSocket

---

### Step 16 : 並列実行速度評価
- **Goal** シングル vs 並列 (例 10 戦) の 1 エピソード処理速度を比較  
- **Done** 理想スケーリングに近い時間短縮・CPU 分散  
- **Test** `time.time()` で計測しログ出力  
- **Tech** Python `time` / logging

---

### Step 17 : 長時間安定性テスト
- **Goal** 100 連戦・数時間連続実行でメモリリーク／接続切れが無いか検証  
- **Done** エラー・ハング無し、メモリ使用量が安定  
- **Test** OS モニタでメモリ推移観察、再接続ログが無いことを確認  
- **Tech** 長時間ベンチ／手動監視

---

### Step 18 : ドキュメント更新
- **Goal** `docs/M6_setup.md` に環境構築・Showdown 起動・並列実行方法などを記載  
- **Done** ドキュメントどおりに再現可能  
- **Tech** Markdown

---

### Step 19 : M6 完了レビュー
- **Goal** 全テスト PASS、100 連戦 & 並列実行 OK、レビュー承認で M6 完了  
- **Done** 「様々なモデル同士の自己対戦が安定・高速に実行可能」な環境を実証  
- **Test** コードリーディング・動作デモでチェックリスト合格  
- **Tech** 総合レビュー

---

## 参考リポジトリ・仕様書
- [`PokemonEnv_Specification.md`](https://github.com/TKG1005/Maple/blob/af7948de455ed5f2d09222c90d3958f4f9e0f771/docs/AI-design/PokemonEnv_Specification.md)  
- [`RLAgent.py`](https://github.com/TKG1005/Maple/blob/af7948de455ed5f2d09222c90d3958f4f9e0f771/src/agents/RLAgent.py)  
- [`pokemon_env.py`](https://github.com/TKG1005/Maple/blob/af7948de455ed5f2d09222c90d3958f4f9e0f771/src/env/pokemon_env.py)

