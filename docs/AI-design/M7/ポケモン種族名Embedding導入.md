# ポケモン種族名Embedding導入 実装手順

## 概要

ポケモンの種族情報を効率的にニューラルネットワークで学習するため、Embedding層を導入します。

### 主な仕様
- ポケモンの名前から全国図鑑No.（0=unknown）に変換
- 状態空間にはidentityでNo.（int）を埋め込む
- 埋込ベクトルは32次元
- 初期化時：先頭6次元は種族値を正規化して埋め、残りは小さな乱数
- 埋込ベクトルは学習時に最適化
- 種族値ベクトルそのものは状態空間には入れない

## 現在の状況分析 (2025-07-12)

### ✅ 既に実装済み
1. **Species Mapping**: `src/utils/species_mapper.py` 完備
   - ポケモン名→図鑑No.変換機能実装済み
   - 1025匹対応、unknownは0で処理

2. **State Space Integration**: `config/state_spec.yml` 対応済み
   - `my_team[0-5].species_id` と `opp_team[0-5].species_id` 実装済み
   - 合計12個のspecies_id特徴量が状態空間に含まれる
   - 現在の状態空間次元: 1136

3. **Pokemon Stats Data**: `config/pokemon_stats.csv` 完備
   - No, name, HP, atk, def, spa, spd, spe, type1, type2, abilities
   - 全1025匹の完全データ

### ✅ 実装完了済み (2025-07-12)
1. **Embedding Layer Architecture**: `src/agents/embedding_networks.py` 実装完了
   - EmbeddingPolicyNetwork/EmbeddingValueNetwork実装済み
   - Species ID抽出・Embedding統合・特徴量結合機能完備
   - freeze_base_stats機能によるgradient masking実装

2. **Base Stats Initialization**: `src/agents/embedding_initializer.py` 実装完了
   - 種族値による重み初期化機能実装済み
   - pokemon_stats.csvから1025匹の種族値データ自動読み込み
   - 正規化・Embedding重み設定・乱数初期化完備

3. **Network Factory Integration**: `src/agents/network_factory.py` 拡張完了
   - "embedding"ネットワークタイプ対応済み
   - EmbeddingPolicyNetwork/EmbeddingValueNetwork作成機能実装

4. **Configuration System**: `config/train_config.yml` 統合完了
   - embedding_configセクション追加済み
   - embed_dim=32, vocab_size=1026, freeze_base_stats設定対応
   - species_indices自動検出機能実装

### 🧪 テスト・検証完了済み
1. **Unit Tests**: `tests/test_embedding_networks.py` 完備
   - 17個のテストケース実装済み
   - Embedding層動作・重み初期化・gradient masking検証完了
   - freeze_base_stats機能の正常動作確認済み

### 🔬 League Training実装完了 (2025-07-12)
1. **Historical Opponent System**: 破滅的忘却対策実装済み
   - 過去のネットワークスナップショット保持機能
   - uniform/recent/weighted選択アルゴリズム実装
   - train_config.ymlでのLeague Training設定対応

### 📊 並列処理最適化完了 (2025-07-12)
1. **TensorBoard分析結果**: 最適並列数特定完了
   - parallel=5が最適効率 (0.76 battles/sec)
   - parallel=10以上は性能低下確認
   - config/train_config.ymlをparallel=5に最適化

---

## 詳細実装手順

### ステップ1: Embedding Network Architecture
**ファイル**: `src/agents/embedding_networks.py`

```python
class EmbeddingPolicyNetwork(nn.Module):
    def __init__(self, observation_space, action_space, embedding_config):
        # 1. Species IDを抽出する部分の実装
        # 2. Embedding層の定義 (vocab_size=1026, embed_dim=32)
        # 3. 種族値による初期化機能
        # 4. 残りの状態特徴量と結合する機能
```

### ステップ2: Base Stats Integration
**ファイル**: `src/agents/embedding_initializer.py`

```python
class EmbeddingInitializer:
    def initialize_species_embeddings(self, embedding_layer, pokemon_stats_csv):
        # 1. CSVから種族値データ読み込み
        # 2. 種族値を0-1正規化
        # 3. Embedding重みの先頭6次元に設定
        # 4. 残り26次元を小さな乱数で初期化
```

### ステップ3: Network Factory Enhancement
**ファイル**: `src/agents/network_factory.py` 拡張

```python
def create_policy_network(observation_space, action_space, config):
    if config.get("use_species_embedding", False):
        return EmbeddingPolicyNetwork(observation_space, action_space, config)
    # 既存のロジック
```

### ステップ4: Configuration Integration
**ファイル**: `config/train_config.yml` 拡張

```yaml
network:
  type: "embedding"  # 新しいネットワークタイプ
  use_species_embedding: true
  embedding_config:
    embed_dim: 32
    freeze_base_stats: false  # 種族値部分の学習可否
    species_indices: [836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847]  # state vectorでのspecies_idの正確な位置
    vocab_size: 1026  # 0(unknown) + 1025(pokemon)
```

### ステップ5: State Vector Processing
**機能**: 状態ベクトルからspecies_idを抽出し、Embeddingに通す

```python
def forward(self, state_vector):
    # 1. state_vectorから12個のspecies_idを抽出
    # 2. Embedding層に通して各32次元ベクトルを取得
    # 3. 他の特徴量と結合
    # 4. 後続の層に渡す
```

### ステップ6: Testing and Validation
1. **Unit Tests**: Embedding層の動作確認
2. **Integration Tests**: 学習パイプラインでの動作確認
3. **Performance Tests**: 学習速度・収束性の検証

---

## 実装優先度と計画

### Phase 1: Core Implementation (高優先度)
1. **EmbeddingInitializer**: 種族値による初期化機能
2. **EmbeddingPolicyNetwork**: 基本的なEmbedding統合
3. **Network Factory Integration**: 既存システムとの統合

### Phase 2: Advanced Features (中優先度)
1. **EmbeddingValueNetwork**: Value network対応
2. **Enhanced Networks Integration**: LSTM/Attention対応
3. **Configuration Validation**: 設定値検証

### Phase 3: Optimization (低優先度)
1. **Performance Tuning**: メモリ・速度最適化
2. **Advanced Initialization**: より高度な初期化手法
3. **Embedding Analysis Tools**: 学習済みEmbeddingの可視化

### ✅ 実装完了状況 (2025-07-12)
- **Phase 1完了**: ✅ 基本的なEmbedding機能の動作完了
- **Phase 2完了**: ✅ 全ネットワークタイプでのEmbedding対応完了  
- **Phase 3実装**: 🔄 最適化とツール整備（League Training・並列処理最適化完了）

### 🎯 追加実装完了事項
- **League Training実装**: 破滅的忘却対策の実装完了
- **並列処理最適化**: TensorBoard分析による最適並列数特定
- **MPS対応**: Apple Metal GPU対応とCPUフォールバック機能
- **Configuration統合**: train_config.ymlへの全設定統合完了

---

## 期待される効果

1. **効率的な学習**: 種族値の事前知識を活用した初期化により、学習の高速化
2. **汎化性能向上**: 未知のポケモンの組み合わせに対する対応力向上
3. **特徴量削減**: 個別の種族値特徴量を削除でき、状態空間の次元削減