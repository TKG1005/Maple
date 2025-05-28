# Maple Project アーキテクチャ概要

## 1. システム全体像

```
+------------------+        WebSocket       +-----------------------+
|  RL Agent        | <--------------------> |  PokemonEnv (Gym)     |
|  (maple.agents)  |                       |  - StateObserver      |
+------------------+                       |  - ActionHelper       |
        ^                                   +----------+-----------+
        |   強化学習ループ (training)                   |
        |                                            poke-env API
        |                                            (HTTP/WS)
        v                                   +----------+-----------+
+------------------+        Local battle    |  Showdown Host       |
|  Replay / Logs   | <--------------------> |  (maple.hosts)       |
+------------------+                       +-----------------------+
```

## 2. フォルダ & ファイル構成

```
Maple/
├── README.md
├── architecture.md      # ← このドキュメント
├── pyproject.toml       # Poetry 推奨。依存: poke-env, gymnasium 等
├── requirements.txt     # pip 使用時
├── .env.example         # APIキーや環境変数のテンプレ
├── src/
│   └── maple/
│       ├── __init__.py
│       ├── config.py            # 共通設定 & 型定義
│       ├── utils/
│       │   ├── logger.py        # ログ設定 (rich 推奨)
│       │   └── registry.py      # 簡易 DI / Factory
│       ├── envs/
│       │   ├── pokemon_env.py   # Gymnasium.Env 実装
│       │   ├── state_observer.py
│       │   └── action_helper.py
│       ├── hosts/
│       │   └── showdown_host.py # poke‑env をラップ
│       ├── agents/
│       │   ├── base_agent.py
│       │   ├── random_agent.py
│       │   └── dqn_agent.py     # 今後実装
│       ├── training/
│       │   ├── train_random.py
│       │   └── train_rl.py
│       ├── evaluation/
│       │   └── evaluate_agent.py
│       ├── data/
│       │   └── parties/
│       │       ├── sample_party.json
│       │       └── ...
│       └── tests/
│           ├── conftest.py
│           ├── test_env_basic.py
│           ├── test_state_observer.py
│           └── test_action_helper.py
├── notebooks/
│   └── exploration.ipynb
├── scripts/
│   ├── start_training.sh
│   └── run_tests.sh
└── docs/
    └── spec.md
```

### 主要ディレクトリの責務

* **envs/**: Gymnasium 互換環境。状態/行動空間をここで統一管理。
* **hosts/**: poke‑env を通じて Showdown! サーバと通信。ローカル or リモートの両対応。
* **agents/**: 行動戦略を実装。強化学習モデルやルールベース。
* **training/**: エージェントを学習させるスクリプト群。設定読み込み → ループ実行 → チェックポイント保存。
* **evaluation/**: モデル性能評価・対戦リプレイ生成。
* **utils/**: 横断的に使うユーティリティ。
* **data/**: パーティ定義やログ、チェックポイント保存先（.gitignore 推奨）。
* **tests/**: pytest ベースのユニットテスト。CI でも実行。

## 3. 主要モジュール詳細

### 3.1 pokemon\_env.py

```python
class PokemonEnv(gymnasium.Env):
    def __init__(self, opponent, observer, action_helper, format="gen9ou"):
        ...
    def step(self, action_index):
        ...
    def reset(self, *, seed=None, options=None):
        ...
```

* **observation\_space**: `spaces.Box(low=0, high=1, shape=(N,), dtype=np.float32)`
* **action\_space**: `spaces.Discrete(A)`

### 3.2 state\_observer.py

* `observe(battle) -> np.ndarray`: Battle から状態ベクトル生成。
* 拡張に備え Strategy パターン採用。

### 3.3 action\_helper.py

* `index_to_order(battle, action_index) -> BattleOrder`
* 有効行動のマスクも提供し、illegal action を防止。

### 3.4 showdown\_host.py

* poke‑env `LocalhostServer` or `Player` をカプセル化。
* 非同期 (`asyncio`) を用いて複数バトル同時実行に対応。

### 3.5 agents/\*

* **RandomAgent**: `act(state) -> int` ただの `env.action_space.sample()`
* **DQNAgent** (予定): PyTorch Lightning で実装。ネットワークは 2層 MLP + Residual。

## 4. データフロー

1. **Agent** が `PokemonEnv.step()` に行動 index を送る。
2. **PokemonEnv** が **ActionHelper** を使い BattleOrder を生成し **poke‑env** へ送信。
3. **poke‑env** が **Showdown Host** 経由で内部シミュレータに伝える。
4. シミュレータから返った **Battle** オブジェクトを **StateObserver** が加工 → 次の状態ベクトル。
5. **PokemonEnv** が報酬・終了判定を行い Agent に返す。

## 5. 開発ワークフロー

```bash
# 1. 環境セットアップ
poetry install
# 2. テスト
pytest
# 3. ランダムエージェントで動作検証
python -m maple.training.train_random --episodes 10
```

* `.env` でポート番号やデバッグフラグを設定可能。

## 6. 強化学習アルゴリズム実装指針

* 初期は **RandomAgent** で環境安定性確認。
* その後 **QLearning** (離散) → **DQN** → **Policy Gradient** 系へ拡張。
* 複数プロセス並列学習には **Ray** or **PettingZoo + SuperSuit** を検討。

## 7. テスト & QA

* pytest + coverage。`tests/` では mock Battle を用意し poke‑env をスタブ可能にする。
* GitHub Actions で `pytest -n auto` を実行し、flake8 & mypy チェック。

## 8. CI / CD

* `pre-commit` で black, isort, flake8。
* GitHub Actions で Lint → Test → Build Docker image (optional)。

## 9. ドキュメント生成

* **docs/** 以下に仕様書や API リファレンス (pdoc) を配置。
* プロジェクトサイト生成は mkdocs-material を予定。

## 10. 今後の拡張ポイント

* 自己対戦 (self-play) を促進するための **League** 管理クラス。
* Switch 実機キャプチャ対応 (M9) 用の Vision モジュール (`maple.vision`).
* ハイパーパラメータ最適化 (Optuna)。

---

> 本ドキュメントは Maple Project (Machine Assisted Pokémon Learning Environment) のアーキテクチャ設計書です。今後の改修に合わせて随時更新してください。
