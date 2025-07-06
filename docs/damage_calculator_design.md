# ダメージ計算モジュール設計計画書

## 1. 概要

本ドキュメントは、「ダメージ計算モジュール要求定義書」に基づき、モジュールの具体的な設計を定義する。
メンテナンス性と拡張性を考慮し、責務を明確に分離したクラスベースの設計を行う。

## 2. 全体構成

モジュールは、計算ロジックを担当する `DamageCalculator` クラスと、計算に必要な静的データを管理するデータファイル群で構成される。

```
src/
└── damage/
    ├── __init__.py
    ├── calculator.py      # DamageCalculatorクラスを定義
    └── data_loader.py     # 静的データを読み込む責務を持つ

config/
├── pokemon_stats.csv    # ポケモンの種族値・タイプ・特性データ
├── moves.csv            # 技の威力・タイプ・分類・追加効果データ
└── type_chart.csv       # タイプ相性表データ
```

## 3. クラス設計

### 3.1. `DamageCalculator` クラス (`calculator.py`)

ダメージ計算の主たるロジックを実装するクラス。

#### 3.1.1. プロパティ

- `pokemon_data`: 全ポケモンのデータ（`DataLoader`から受け取る）
- `move_data`: 全技のデータ（`DataLoader`から受け取る）
- `type_chart`: タイプ相性表データ（`DataLoader`から受け取る）

#### 3.1.2. メソッド

- `__init__(self, data_loader)`
  - `DataLoader`のインスタンスを受け取り、各種データをプロパティに格納する。

- `calculate_damage_range(self, attacker, defender, move, field_state)`
  - **役割:** 人間向けのダメージ範囲計算（要求定義 2.2）
  - **引数:**
    - `attacker`: 攻撃側ポケモンの状態オブジェクト
    - `defender`: 防御側ポケモンの状態オブジェクト
    - `move`: 使用する技オブジェクト
    - `field_state`: 場の手状態オブジェクト
  - **戻り値:** ダメージ範囲、HP割合、確定数、計算内訳を含む辞書。

- `simulate_move_effect(self, attacker, defender, move, field_state)`
  - **役割:** AIエージェント向けの単一結果シミュレーション（要求定義 2.3）
  - **引数:** `calculate_damage_range` と同様。
  - **戻り値:** 命中判定、実ダメージ、急所判定、追加効果の結果を含む辞書。

### 3.2. `DataLoader` クラス (`data_loader.py`)

CSV/YAMLファイルから静的データを読み込み、プログラムで扱いやすい形式に変換する責務を持つクラス。

#### 3.2.1. メソッド

- `load_pokemon_stats(self, file_path)`: `pokemon_stats.csv` を読み込む。
- `load_moves(self, file_path)`: `moves.csv` を読み込む。
- `load_type_chart(self, file_path)`: `type_chart.csv` を読み込む。

## 4. データファイル設計

### 4.1. `pokemon_stats.csv`

| name | hp | attack | defense | sp_attack | sp_defense | speed | type1 | type2 | ability1 | ability2 |
| :--- | :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ピカチュウ | 35 | 55 | 40 | 50 | 50 | 90 | でんき | | せいでんき | ひらいしん |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 4.2. `moves.csv`

| name | type | category | power | accuracy | pp | effect_type | effect_prob |
| :--- | :--- | :--- | ----: | -------: | -: | :--- | ----------: |
| 10まんボルト | でんき | 特殊 | 90 | 100 | 15 | まひ | 10 |
| ... | ... | ... | ... | ... | ... | ... | ... |

### 4.3. `type_chart.csv`

| attacking_type | defending_type | multiplier |
| :--- | :--- | ---------: |
| ノーマル | いわ | 0.5 |
| ノーマル | はがね | 0.5 |
| ほのお | くさ | 2.0 |
| ... | ... | ... |
