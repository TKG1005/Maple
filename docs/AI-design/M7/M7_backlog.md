# Mapleプロジェクト M7 バックログ＋背景・戦略補足版


## 概要・現状の課題

Mapleプロジェクトではポケモン対戦AIの強化学習による自律学習を進めてきたが、以下の主要課題が存在する：

- **モデル容量不足**：NNが浅く複雑な戦術を十分学習できない可能性
- **スパース報酬**：勝敗のみの報酬で学習効率が低い（クレジット割当困難）
- **探索の弱さ**：エージェントが多様な戦略を試しにくい
- **対戦相手の多様性不足**：自己対戦のみで汎用性に欠ける
- **部分観測・長期戦略未対応**：現状は履歴や隠れ情報を活かせていない

これらは、先行AI事例や理論的知見（部分観測ゲーム・報酬シェイピング・自己対戦とカリキュラム学習等）に照らしても、強化学習性能・実戦対応力のボトルネックである。

---

## M7 改善戦略（全体方針）

1. **報酬設計の改善**（R系タスク）  
   → 勝敗以外にも撃破・HP変化などで段階的に学習信号を与え、効率的に方策改善を促す。
2. **対戦相手の多様化**（B系タスク）  
   → ランダムbotやルールベースbotとの対戦導入、自己対戦と比率調整することで戦術多様性に耐性あるAIにする。
3. **ネットワーク構造拡張**（N系タスク）  
   → 全結合層の多層化やLSTM、Attention層で表現力と履歴対応力を高める。
4. **探索戦略強化**（E系タスク）  
   → エントロピーボーナスやε-greedyを拡張し、行動の偏り・局所解への陥り込みを防ぐ。
5. **評価と可視化**（V系タスク）  
   → 報酬の内訳や行動多様性をログ・TensorBoardで可視化し、課題抽出と進捗監視を容易に。
6. **アルゴリズム追加・比較**（A系タスク）  
   → A2C実装やハイパラ検索でアルゴリズム面からも強化方針を探る。
7. **CI・自動化**（C系タスク）  
   → テスト・PR管理・スタイル統一を通じて開発の健全性も担保。

---

## なぜこの順序か

- **報酬シェイピングが最大の効率改善ポイント**：学習初期から即座に改善効果を発揮する。
- **対戦相手多様化はAIの汎用性向上に直結**：固定自己対戦では学べない戦略幅を身につけるため必須。
- **NN・探索・評価は基盤強化・性能可視化の役割**：学習安定化と分析効率アップ。
- **部分観測・履歴活用（LSTM等）は、長期的に勝てるAIへの布石**：これを意識し実装に反映。

---

## 先行事例と照らした設計意図

- 先行AI（poke-envサンプルなど）は2層MLP＋報酬シェイピング＋多様な訓練相手で高速な性能向上を実現している。
- 逆に、スパース報酬＋浅いNN＋自己対戦限定では伸び悩む事例が多い。
- Maple M7では「理論・実績に基づく強化学習の本筋」を押さえたうえで実装タスクを小分割し、開発・検証・自動化の効率も同時に追求する。

---

## 前提   
* 1 タスク = “実装 → 検証（短時間学習）→ ログ確認／簡易分析” までをワンセットで完了  
* すべて **Python 3 + PyTorch 2.x + Gymnasium** 準拠。  

---

## 1. 報酬シェイピング

| ID  | タスク                          | 対象ファイル               | 技術要件／検証 |
|-----|-------------------------------|----------------------------|----------------|
| R-1 | `HP差スコア` 報酬クラス追加        | `rewards/hp_delta.py`      | PyTorch で ΔHP を逐ターン計算。UnitTest で数値確認。短時間学習でエピソード平均報酬上昇を確認。 |
| R-2 | 報酬合成マネージャ              | `rewards/composite.py`, `config/reward.yaml` | YAML で各係数を切替。TensorBoard で sub-reward 値をプロット。 |
| R-3 | `撃破/被撃破 ±1` 実装           | `rewards/knockout.py`      | エージェント側・敵側の faint 検知ロジック追加。ログに内訳出力。 |
| R-4 | 状態異常付与時の報酬クラス追加               | `rewards/status_effect.py`      | 状態異常判定・加点・UnitTest |
| R-5 | 積み技成功時の微加点                       | `rewards/stat_boost.py`         | 能力値上昇検知・加点          |
| R-6 | 交代直後にタイプ有利になった場合の加点        | `rewards/switch_advantage.py`   | 交代判定・タイプ相性計算      |
| R-7 | 毎ターン経過ペナルティ                     | `rewards/turn_penalty.py`       | ターンごとに-0.01付与         |
| R-8 | 最終味方数-相手数で差分報酬                 | `rewards/team_diff.py`          | エピソード終了時に判定・加点   |
| R-9 | 報酬合成管理の強化（優先度・重み制御機能追加） | `rewards/composite.py`          | 各sub-rewardの重みパラメータ  |
| R-10 | 無効行動時のペナルティ            | `rewards/fail_and_immute.py`    | 自分の無効行動にペナルティを与える    |

### 報酬設計・仕様

#### HP差報酬（R-1）
- **自分/相手、ダメージ/回復を分離して計算・ログ**  
  - 自分のHP減少：ペナルティ  
  - 自分のHP回復：ボーナス  
  - 相手のHP減少：ボーナス  
  - 相手のHP回復：ペナルティ  
- 各値は「ターンごとのHP差分 / 最大HP（正規化）」で報酬値を算出  
- 各ターンで「ごく小さい報酬/ペナルティ」を与える  
- **バトル終了時**、たとえば「自分の全体HPが半分以上残っていれば追加ボーナス」など条件付きボーナスも実装

#### 報酬合成・係数管理（R-2）
- HP差報酬と撃破報酬は**初期状態では同じ重み**。各サブ報酬は**YAML等の設定ファイルで係数変更可能**  
- 各サブ報酬の係数や有効/無効化を個別に切り替え可能  
- TensorBoard等で**各サブ報酬ごとの値を個別にプロット**し、学習内訳を可視化

#### 撃破/被撃破報酬（R-3）
- ポケモン撃破時（faint）に**+1**、被撃破時に**-0.5**（初期値。YAMLで係数調整可）  
- チーム単位で管理し、「どのポケモンが倒したか」は区別しない  
- ログには各サブ報酬ごとの値を出力

#### 実装上の留意点
- 全報酬ロジックを`composite.py`等で一元管理、YAMLの内容で各報酬を合成
- コード、ログ、グラフすべて「サブ報酬ごとに見える化」しやすい設計を推奨

#### 初期パラメータ例（YAML想定）
```yaml
reward:
  hp_self_damage:    -0.01
  hp_self_heal:      +0.01
  hp_enemy_damage:   +0.01
  hp_enemy_heal:     -0.01
  faint_enemy:       +1.0
  faint_self:        -0.5
  win_bonus:         +10.0  # 例: 勝利時 or HP条件達成時
```

#### CompositeReward の使い方

`CompositeReward` クラスは複数のサブ報酬をまとめて管理するマネージャです。
`PokemonEnv` を `reward="composite"` で初期化し、
`reward_config_path` に YAML ファイルを渡すことで利用できます。
`train_selfplay.py` では次のように指定します。

```bash
python train_selfplay.py --reward composite --reward-config config/reward.yaml
```

YAML ファイルの例を以下に示します。各サブ報酬は `enabled` で有効/無効を切り替え、
`weight` で重みを調整します。

```yaml
rewards:
  hp_delta:
    weight: 1.0
    enabled: true
  knockout:
    weight: 1.0
    enabled: true
  turn_penalty:
    weight: 1.0
    enabled: true
```

任意の報酬を追加する場合は `rewards:` 配下に項目を増やすだけでよく、
コード側は自動的に読み込んで合成します。

---

## 2. 対戦相手の多様化

| ID  | タスク                          | 対象ファイル               | 技術要件／検証 |
|-----|-------------------------------|----------------------------|----------------|
| B-1 | `RandomBot` 実装               | `bots/random_bot.py`       | Gymnasium `Env` 互換ステップ。vs self-play 比率切替 param 追加。 |
| B-2 | `MaxDamageBot` 実装             | `bots/max_damage_bot.py`   | poke-env の同名ロジックを移植。5 戦評価で勝率差をログ。 |
| B-3 | 学習相手スケジューラ            | `train/opponent_pool.py`   | `--opponent_mix random:0.3,max:0.3,self:0.4` 形式パース。 |
| B-4 | 複数チームファイルのランダム使用  | `teams/team_loader.py`, `config/teams.yaml` | Pokemon Showdown形式チーム解析。ディレクトリ一括読み込み。学習・評価時のランダム選択機能。 |

---

## 3. ネットワーク拡張

| ID  | タスク                          | 対象ファイル               | 技術要件／検証 |
|-----|-------------------------------|----------------------------|----------------|
| N-1 | MLP 2 層化 (`128→256→128`)     | `agents/networks.py`       | 既存重みロード互換を維持（キー差し替え）。10 k step で学習曲線比較。 |
| N-2 | LSTM ヘッダ追加（オプション）     | `agents/networks.py`       | 1 層 LSTM + 128 hidden。`ObsStacker` から履歴供給。 |
| N-3 | アテンション試験フック           | `agents/attention.py`      | Multi-head self-attn を味方/敵特徴ごとに適用。ベンチマークのみ。 |

---

## 4. 探索戦略強化

| ID  | タスク                          | 対象ファイル               | 技術要件／検証 |
|-----|-------------------------------|----------------------------|----------------|
| E-1 | PPO エントロピー係数 config 化   | `train/ppo_trainer.py`     | `--entropy_coef 0.01` CLI。エントロピー平均を TensorBoard 出力。 |
| E-2 | ε-greedy wrapper 実装           | `agents/action_wrapper.py` | ε を線形or指数減衰 (1→0.1) オプション。ランダム行動(探索)率を毎試合ログ。 |

---

## 5. 評価 & ロギング

| ID  | タスク                          | 対象ファイル               | 技術要件／検証 |
|-----|-------------------------------|----------------------------|----------------|
| V-1 | TensorBoard スカラー整理         | `eval/tb_logger.py`        | `win_rate`, `avg_reward`, 各 sub-reward, `entropy` を統一命名。 |
| V-2 | CSV エクスポートユーティリティ     | `eval/export_csv.py`       | 学習終了時に `runs/YYYYMMDD/metrics.csv` を出力。 |
| V-3 | 行動多様性メトリクス              | `eval/diversity.py`        | 技選択分布 KL 距離を算出しグラフ化。 |

---

## 6. アルゴリズム追加

| ID  | タスク                          | 対象ファイル               | 技術要件／検証 |
|-----|-------------------------------|----------------------------|----------------|
| A-1 | A2C 実装                        | `train/a2c_trainer.py`     | 共有ネット＋同期 actor。既存 `ActorCritic` を再利用。 |
| A-2 | ハイパラ検索スクリプト            | `train/hparam_search.py`   | Optuna で A2C vs PPO を 3 試行ずつ回し、勝率を比較。 |

---

## 7. CI / 自動化

| ID  | タスク                          | 対象ファイル               | 技術要件／検証 |
|-----|-------------------------------|----------------------------|----------------|
| C-1 | GitHub Actions スモーク           | `.github/workflows/test.yml` | `pytest -q` & 1 k step 学習で loss nan しないこと。 |
| C-2 | Codex/LLM 用 `TODO.md`           | `docs/TODO_M7.md`          | タスク一覧を機械可読 (- [ ] 形式) に列挙し PR に紐付け。 |
| C-3 | Pre-commit Black + ruff          | `.pre-commit-config.yaml`  | 自動フォーマット・Lint 通過を必須化。 |

---

### 推奨実施順

1. **R-系** 報酬設計の見直し（中間報酬追加／シェイピング） → 学習効率に即影響  
2. **B-系** 対戦相手多様化（bot導入、自己対戦との組み合わせ） 
3. **N-系** ニューラルネット構造の拡張（2層化・LSTM等）
4. **E-系** 探索戦略強化（エントロピー係数増加、ε-greedy）
5. **V-系** 評価・ログ可視化
6. **A-系** アルゴリズムの追加実験（A2C等）
7. **C-系** 以上を進めつつ、常に勝率・報酬平均・行動多様性を観測し、フィードバックループを短く回す

---

#### 各タスクの進め方テンプレ

1. **ブランチ作成** `feature/<ID>-short-desc`  
2. コード実装 + **pytest**（新規/更新）  
3. `python train/quick_run.py --steps 1000 --config config/m7.yaml`  
4. TensorBoard / CSV を確認し **before→after** 差分を記録  
5. PR 作成（結果スクリーンショット添付）  
6. **レビュー & マージ**  
7. 次タスクへ

---

### 備考

1. すべてPython 3, PyTorch, Gymnasium準拠。  
2. 各施策の**動機・戦略・他事例との対応**を明示し、単なるToDoではなく「なぜやるか」も記録している点が特徴。
3. 実装に際しては、現状のリポジトリ構成や既存CI設計を活かす。


## 実装進捗

- [x] R-1 HP差スコア報酬クラス追加
- [x] R-2 報酬合成マネージャ
- [x] R-3 撃破/被撃破 ±1 実装
- [ ] R-4 状態異常付与時の報酬クラス追加
- [ ] R-5 積み技成功時の微加点
- [ ] R-6 交代直後にタイプ有利になった場合の加点
- [x] R-7 毎ターン経過ペナルティ
- [x] R-8 最終味方数-相手数で差分報酬
- [ ] R-9 報酬合成管理の強化
- [x] R-10 無効行動時のボーナス・ペナルティ
- [x] B-1 RandomBot 実装
- [x] B-2 MaxDamageBot 実装
- [x] B-3 学習相手スケジューラ
- [x] B-4 複数チームファイルのランダム使用
- [x] N-1 MLP 2 層化（基本・LSTM・Attentionネットワーク実装）
- [x] N-2 LSTM ヘッダ追加（隠れ状態管理とシーケンシャル学習対応）
- [x] N-3 アテンション試験フック（Multi-head Attentionネットワーク実装）
- [x] E-1 PPO エントロピー係数 config 化（config/train_config.ymlで実装済み + TensorBoard出力実装完了）
- [ ] E-2 ε-greedy wrapper 実装
- [ ] V-1 TensorBoard スカラー整理
- [ ] V-2 CSV エクスポートユーティリティ
- [ ] V-3 行動多様性メトリクス
- [ ] A-1 A2C 実装
- [ ] A-2 ハイパラ検索スクリプト
- [ ] C-1 GitHub Actions スモーク
- [x] C-2 Codex/LLM 用 TODO.md（CLAUDE.mdとして実装完了）
- [ ] C-3 Pre-commit Black + ruff

## 新規追加実装 (2025-07-10)

- [x] **設定ファイルシステム**: YAMLベースのパラメータ管理
  - `config/train_config.yml`: テスト・短時間駆動用（10エピソード、混合対戦相手、LSTMネットワーク）
  - `config/train_config_long.yml`: 長時間学習用（1000エピソード、セルフプレイ、Attentionネットワーク）
  - コマンドライン引数の簡素化と設定の再利用性向上

- [x] **勝率ベース対戦相手更新システム**: 効率的なセルフプレイ学習
  - 勝率閾値（デフォルト60%）による条件付き対戦相手更新
  - 対戦相手スナップショット管理
  - 過度なネットワークコピー削減による学習効率向上
  - 設定可能な勝率閾値と監視ウィンドウ

- [x] **LSTM隠れ状態管理修正**: シーケンシャル学習の実現
  - バッチ処理対応の隠れ状態管理
  - エピソード境界での隠れ状態リセット
  - 学習安定性の大幅改善

- [x] **ネットワーク互換性修正**: 全ネットワーク対応
  - 基本・LSTM・Attentionネットワークの統一インターフェース
  - 条件分岐による forward メソッド互換性確保

## 新規追加実装 (2025-07-12)

### ポケモン種族名Embedding機能

- [x] **EmbeddingInitializer実装**: 種族値による初期化機能
  - `src/agents/embedding_initializer.py`: 1025匹の種族値データ自動読み込み
  - 正規化・Embedding重み設定・乱数初期化機能
  - CSVから種族値データを抽出して先頭6次元に設定

- [x] **EmbeddingPolicyNetwork/ValueNetwork実装**: ニューラルネットワーク統合
  - `src/agents/embedding_networks.py`: Policy/Value両ネットワーク対応
  - Species ID抽出・Embedding統合・特徴量結合機能
  - 32次元Embedding（6次元種族値＋26次元学習可能）

- [x] **freeze_base_stats機能**: 種族値次元の勾配制御
  - gradient maskingによる先頭6次元の勾配凍結
  - register_hook()を使用した効率的な実装
  - 種族値情報を保持しつつ学習を制御

- [x] **Network Factory統合**: 既存システムとの完全統合
  - `src/agents/network_factory.py`: "embedding"タイプサポート
  - 既存の基本・LSTM・Attentionネットワークとの互換性維持

- [x] **Configuration統合**: YAML設定でのEmbedding制御
  - `config/train_config.yml`: embedding_configセクション追加
  - embed_dim、vocab_size、freeze_base_stats設定対応
  - species_indices自動検出機能

- [x] **包括的なUnit Test**: 17個のテストケース実装
  - `tests/test_embedding_networks.py`: 全機能の動作検証
  - 重み初期化・gradient masking・統合動作確認

### League Training機能（破滅的忘却対策）

- [x] **Historical Opponent System**: 過去のネットワーク保持機能
  - 最大5個の過去ネットワークスナップショット管理
  - 固定対戦相手による破滅的忘却の防止
  - train_selfplay.pyに統合実装

- [x] **選択アルゴリズム実装**: 3つの履歴選択方法
  - uniform: 全履歴から均等確率で選択
  - recent: 新しい履歴を優先的に選択
  - weighted: 新しさに比例した重み付け選択

- [x] **Configuration対応**: League Training設定
  - historical_ratio: 履歴対戦の比率（デフォルト0.3）
  - max_historical: 保持する履歴数（デフォルト5）
  - selection_method: 選択アルゴリズム指定

### 並列処理最適化

- [x] **TensorBoard分析実行**: 並列効率の詳細分析
  - parallel=5,10,20,30での時間あたりバトル数測定
  - Pokemon Showdownサーバーのボトルネック特定
  - 並列度増加による性能低下の定量化

- [x] **最適設定特定**: parallel=5が最高効率
  - 0.76 battles/sec（システム全体スループット）
  - 並列度10以上では逆に性能低下（-33%〜-73%）
  - エピソード完了時間が並列度に比例して増加

- [x] **設定ファイル最適化**: 実運用設定への反映
  - `config/train_config.yml`: parallel=5に最適化
  - 効率的な学習のための推奨設定確立
