# ポケモン技ベース表の前処理～Embedding変換 実装計画

---

## 概要
技データのベース表（例：moves.csv）を機械学習で使える数値ベクトル（Embedding）に変換するまでの**実装手順**をまとめます。  
※Python（pandas, scikit-learn, numpy, sentence-transformers 等）での処理を前提にしています。

---

## 前提：ベース表の例

| move_id | name     | category | type      | power | accuracy | pp | priority | ... |
|---------|----------|----------|-----------|-------|----------|----|----------|-----|
| 1       | たいあたり | Physical | Normal    | 40    | 1.0      | 35 | 0        | ... |
| 2       | でんじは   | Status   | Electric  | 0     | 0.90     | 20 | 0        | ... |
| ...     | ...      | ...      | ...       | ...   | ...      | ...| ...      | ... |

---

## 実装フロー

### 1. データ読み込み

```python
import pandas as pd

moves = pd.read_csv('moves.csv')

2. 欠損値・型の確認＆補正
必要なカラムが揃っているか

欠損値の埋め・型変換

python
コピーする
編集する
# 例: 欠損値を0やFalseで埋める
moves['power'] = moves['power'].fillna(0).astype(float)
moves['accuracy'] = moves['accuracy'].fillna(1.0).astype(float)
moves['priority'] = moves['priority'].fillna(0).astype(int)
# 他のブール/フラグ類も必要に応じて0/1で埋める

3. カテゴリ変数のエンコーディング（one-hot等）
例：type, category

python
コピーする
編集する
# タイプ（例: 18種）をone-hot化
type_ohe = pd.get_dummies(moves['type'], prefix='type')

# カテゴリ（物理/特殊/変化）もone-hot化
cat_ohe = pd.get_dummies(moves['category'], prefix='category')

# 結合
moves_enc = pd.concat([moves, type_ohe, cat_ohe], axis=1)

4. 数値スケーリング・正規化
必要に応じて、MinMaxScalerやMax値割りで[0,1]にスケール

python
コピーする
編集する
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
for col in ['power', 'accuracy', 'pp', 'priority']:
    moves_enc[col + '_scaled'] = scaler.fit_transform(moves_enc[[col]])
（または分母で割るなど単純なスケールも可）

5. ブール値（フラグ）の処理
0/1のfloatに統一

python
コピーする
編集する
for flag_col in ['ohko_flag', 'contact_flag', 'sound_flag', 'protectable', 'substitutable', ...]:
    moves_enc[flag_col] = moves_enc[flag_col].astype(float)

6. 追加効果（secondary_effects等）が別表の場合
ピボット・集約して技ごとに特徴ベクトル化（必要に応じて）

例: 各effect_idごとに1列（0/1 or 確率値）

python
コピーする
編集する
# secondary_effects.csvの読み込み
effects = pd.read_csv('secondary_effects.csv')

# move_idごとに効果ごとone-hotまたはagg
pivot = effects.pivot_table(index='move_id',
                            columns='effect_id',
                            values='probability',
                            aggfunc='max',
                            fill_value=0)
moves_enc = moves_enc.merge(pivot, left_on='move_id', right_index=True, how='left').fillna(0)

7. 自然言語説明のベクトル化
図鑑説明/テキスト部分を事前学習済みSentence-Transformer等でエンコード

PCAなどで圧縮（256次元など）

python
コピーする
編集する
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np

# モデル準備
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 技説明カラムが 'base_desc' とする
descs = moves_enc['base_desc'].fillna('').tolist()
desc_vecs = model.encode(descs, normalize_embeddings=True)

# PCAで圧縮
pca = PCA(n_components=256)
desc_vecs_reduced = pca.fit_transform(desc_vecs)

# DataFrameに戻す
desc_df = pd.DataFrame(desc_vecs_reduced, columns=[f'desc_emb_{i}' for i in range(256)])
moves_enc = pd.concat([moves_enc, desc_df], axis=1)

8. 最終ベクトルの組み立て
不要カラム（IDや生テキスト等）を除外

機械学習・Embedding入力用の1行＝1技のベクトルが完成

python
コピーする
編集する
feature_cols = (
    list(type_ohe.columns) +
    list(cat_ohe.columns) +
    [col + '_scaled' for col in ['power', 'accuracy', 'pp', 'priority']] +
    [flag_col for flag_col in ['ohko_flag', 'contact_flag', 'sound_flag', 'protectable', 'substitutable', ...]] +
    list(pivot.columns) +
    [f'desc_emb_{i}' for i in range(256)]
)

X = moves_enc[feature_cols].values  # (技数, 特徴数)

9. Embeddingモデルへの入力・利用
そのままAIモデル/RLエージェントへ入力

または教師なしクラスタリング・類似検索等にも利用可能

