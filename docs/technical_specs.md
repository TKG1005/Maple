# Maple プロジェクト 技術仕様

## プロジェクト概要
Maple は `poke_env` と `gymnasium` を利用してポケモン対戦 AI を実装する Python プロジェクトです。設定ファイル類は `config/`、主要コードは `src/`、テストは `test/` に配置されています。

## ディレクトリ構成
- `config/` : 状態定義 YAML や行動マッピングなどの設定ファイル
- `docs/`   : ドキュメント類
- `src/`
  - `action/` : 行動マスク生成やインデックス変換を行うモジュール
  - `agents/` : `poke_env` ベースのプレイヤークラス群
  - `env/`    : Gymnasium 環境 `PokemonEnv`
  - `state/`  : `StateObserver` など状態取得関連
- `test/` : `pytest` によるユニットテスト

## 依存ライブラリ
`requirements.txt` では `gymnasium`, `poke_env`, `numpy`, `pytest` 等を指定しています。【F:requirements.txt†L1-L31】

## PokemonEnv クラス
`src/env/pokemon_env.py` に実装されている Gymnasium 環境です。主な仕様は以下の通りです。

### 初期化
```python
class PokemonEnv(gym.Env):
    def __init__(
        self,
        opponent_player: Any,
        state_observer: Any,
        action_helper: Any,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.opponent_player = opponent_player
        self.state_observer = state_observer
        self.action_helper = action_helper
        self.rng = np.random.default_rng(seed)
        state_dim = self.state_observer.get_observation_dimension()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(ACTION_SIZE)
```
【F:src/env/pokemon_env.py†L1-L48】
- `opponent_player` などを依存注入する形式です。
- 観測空間は `StateObserver` が返す次元数から作成されます。
- 行動空間は `ACTION_SIZE` (10) を用いた離散空間です。【F:src/action/__init__.py†L4-L7】

### 基本メソッド
- `reset` : まだダミー実装で、初期観測値を返す処理は TODO です。
- `step`  : 行動を受け取ってバトルを 1 ターン進める枠組みのみが用意されています。
- `render` / `close` : デバッグ用の空実装。

## ActionHelper
`src/action/action_helper.py` では現在のバトル状態から行動マスクを生成し、インデックスを `poke_env` の命令に変換するユーティリティを提供します。行動のスロット構成は以下です。
- 0–3 : 通常技
- 4–7 : テラスタル技 (通常技をミラー)
- 8–9 : ポケモン交代
【F:src/action/action_helper.py†L14-L34】【F:src/action/action_helper.py†L142-L175】

## StateObserver
`src/state/state_observer.py` は `config/state_spec.yml` を読み込み、バトルオブジェクトから特徴量を抽出してベクトル化します。エンコーダの種類として `identity`、`onehot`、`linear_scale` があり、観測次元数計算もサポートしています。【F:src/state/state_observer.py†L1-L120】【F:src/state/state_observer.py†L120-L240】

## テスト
`test/` 以下の `test_pokemon_env_step2_4.py` などで環境インスタンス生成や `action_space`/`observation_space` の性質を確認するテストを実施しています。【F:test/test_pokemon_env_step2_4.py†L1-L63】【F:test/test_pokemon_env_step5.py†L1-L72】

---
以上が現時点の Maple プロジェクトおよび `PokemonEnv` の主要な技術仕様です。
